#!/usr/bin/env python3
"""Run the EAA ranker under a labeled evaluation condition (HM / HMT / adaptive)."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from websocietysimulator import Simulator
from websocietysimulator.agent.recommendation_agent import RecommendationAgent
from websocietysimulator.conditions.task_conditions import (
    build_task_contexts,
    get_condition_task_indices,
    load_condition_manifest,
)
from websocietysimulator.evaluation_io import (
    default_metrics_path,
    default_predictions_path,
    save_metrics_result,
    save_predictions,
)
from websocietysimulator.tools.conditional_interaction_tool import ConditionalInteractionTool
from websocietysimulator.tools.semantic_id_tool import SemanticIDTool

from example.EAARecAgent_baseline import make_eaa_agent_class
from example.eaa_evidence_policy import load_policy_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_condition_eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--condition",
        required=True,
        choices=["classic", "cold_start_user", "cold_start_item", "evolving_interest"],
    )
    parser.add_argument(
        "--agent",
        default="eaa",
        choices=["eaa"],
        help="Paper reproduction only supports the EAA ranker.",
    )
    parser.add_argument(
        "--eaa_force_modules",
        default=None,
        help="Comma-separated modules, e.g. H,M (HM) or H,M,S_tool (HMT).",
    )
    parser.add_argument(
        "--eaa_no_sid_merge",
        action="store_true",
        help="Prompt-only HMT: show SID scores but disable merge/fallback.",
    )
    parser.add_argument("--eaa_trust_selection", action="store_true")
    parser.add_argument(
        "--eaa_policy",
        default="quality",
        choices=["quality", "v3_scenario"],
    )
    parser.add_argument(
        "--eaa_policy_config",
        default=None,
        help="JSON from scripts/calibrate_eaa_policy.py.",
    )
    parser.add_argument("--llm_module", default="websocietysimulator.llm")
    parser.add_argument("--llm_class", default="InfinigenceLLM")
    parser.add_argument("--api_key", default=os.environ.get("LLM_API_KEY", "your api_key"))
    parser.add_argument(
        "--ollama_model",
        default="qwen2.5:7b-instruct",
        help="Ollama tag when --llm_class OllamaLLM.",
    )
    parser.add_argument(
        "--artifact_dir",
        required=True,
        help="SID-v2 directory containing item_id_to_sid.pt.",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--predictions_output", default=None)
    parser.add_argument("--predictions_dir", default=os.path.join(ROOT, "outputs"))
    parser.add_argument("--results_dir", default=os.path.join(ROOT, "results"))
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--calibration_config", default=None)
    parser.add_argument(
        "--eval_split",
        choices=["all", "validation", "heldout"],
        default="all",
    )
    parser.add_argument("--simulator_device", default="cpu", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument(
        "--evidence_no_reviews",
        action="store_true",
        help="History ablation: titles only, no review text.",
    )
    parser.add_argument("--enable_threading", action="store_true")
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--no_cache", action="store_true")
    return parser.parse_args()


def task_indices_for_eval_split(indices, *, calibration_config, eval_split):
    if eval_split == "all":
        return indices
    if not calibration_config:
        raise ValueError("--eval_split requires --calibration_config")
    with open(calibration_config, encoding="utf-8") as handle:
        payload = json.load(handle)
    cal_meta = payload.get("calibration", payload)
    key = (
        "validation_task_indices"
        if eval_split == "validation"
        else "heldout_task_indices"
    )
    allowed = set(cal_meta[key])
    filtered = [index for index in indices if index in allowed]
    if not filtered:
        raise ValueError(f"No tasks matched eval_split={eval_split!r}")
    return filtered


def resolve_agent(args):
    force_modules = None
    if args.eaa_force_modules:
        force_modules = [
            chunk.strip()
            for chunk in args.eaa_force_modules.split(",")
            if chunk.strip()
        ]
    policy_config = None
    if args.eaa_policy_config:
        policy_config = load_policy_config(args.eaa_policy_config)
    agent_class = make_eaa_agent_class(
        force_modules=force_modules,
        use_sid_merge_fallback=not args.eaa_no_sid_merge,
        trust_selection=args.eaa_trust_selection,
        condition=args.condition,
        policy_mode=args.eaa_policy,
        policy_config=policy_config,
        no_reviews=args.evidence_no_reviews,
    )
    llm_module = importlib.import_module(args.llm_module)
    llm_class = getattr(llm_module, args.llm_class)
    if args.llm_class == "OllamaLLM":
        llm = llm_class(model=args.ollama_model)
    else:
        llm = llm_class(api_key=args.api_key)
    return agent_class, llm


def maybe_wrap_interaction_tool(simulator: Simulator, condition: str) -> None:
    if condition != "evolving_interest":
        return
    wrapped = ConditionalInteractionTool(simulator.interaction_tool)
    simulator.set_interaction_tool(wrapped)
    if simulator.semantic_id_tool is not None:
        simulator.semantic_id_tool.set_interaction_tool(wrapped)


def build_semantic_id_tool(dataset: str, artifact_dir: str) -> SemanticIDTool:
    artifact_dir = os.path.abspath(artifact_dir)
    marker = os.path.join(artifact_dir, "item_id_to_sid.pt")
    if not os.path.isfile(marker):
        raise FileNotFoundError(
            f"SID artifacts not found at {artifact_dir} (missing item_id_to_sid.pt). "
            "Build with scripts/build_sid_v2.py."
        )
    logger.info("Loading SemanticIDTool from %s", artifact_dir)
    return SemanticIDTool(
        artifact_dirs={dataset: artifact_dir},
        default_source=dataset,
    )


def main() -> None:
    args = parse_args()
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["OLLAMA_NUM_GPU"] = "0"
        args.simulator_device = "cpu"
        print("[device] --device cpu: torch on CPU, Ollama num_gpu=0.")

    manifest = load_condition_manifest(args.manifest)
    metadata = manifest["metadata"]
    dataset = metadata["dataset"]
    track = metadata["track"]
    task_dir = metadata["task_dir"]
    groundtruth_dir = metadata["groundtruth_dir"]

    agent_class, llm = resolve_agent(args)
    if track != 2 or not issubclass(agent_class, RecommendationAgent):
        raise ValueError("This reproduction package only runs Track-2 recommendation.")

    simulator_device = args.simulator_device
    simulator = Simulator(
        data_dir=args.data_dir,
        block_set_dir=os.path.dirname(task_dir),
        device=simulator_device,
        cache=not args.no_cache,
    )
    simulator.set_semantic_id_tool(
        semantic_id_tool=build_semantic_id_tool(dataset, args.artifact_dir),
        default_source=dataset,
    )
    simulator.set_task_and_groundtruth(task_dir=task_dir, groundtruth_dir=groundtruth_dir)

    indices = get_condition_task_indices(manifest, args.condition)
    indices = task_indices_for_eval_split(
        indices,
        calibration_config=args.calibration_config,
        eval_split=args.eval_split,
    )
    contexts = build_task_contexts(manifest, args.condition)
    simulator.set_task_contexts(contexts)
    simulator.filter_tasks(indices)
    maybe_wrap_interaction_tool(simulator, args.condition)
    simulator.set_agent(agent_class)
    simulator.set_llm(llm)

    task_indices = indices[: args.max_tasks] if args.max_tasks is not None else indices
    simulator.run_simulation(
        number_of_tasks=args.max_tasks,
        enable_threading=args.enable_threading,
        max_workers=args.max_workers,
    )
    evaluation_results, error_log = simulator.evaluate()

    run_metadata = {
        "agent": args.agent,
        "condition": args.condition,
        "dataset": dataset,
        "track": track,
        "artifact_dir": os.path.abspath(args.artifact_dir),
        "task_count": len(task_indices),
        "task_indices": task_indices,
        "eval_split": args.eval_split,
        "calibration_config": (
            os.path.abspath(args.calibration_config) if args.calibration_config else None
        ),
        "manifest": os.path.abspath(args.manifest),
        "device": args.device,
        "evidence_no_reviews": bool(args.evidence_no_reviews),
        "llm_class": args.llm_class,
        "ollama_model": args.ollama_model if args.llm_class == "OllamaLLM" else None,
        "simulator_device": args.simulator_device,
        "eaa_force_modules": args.eaa_force_modules,
        "eaa_no_sid_merge": args.eaa_no_sid_merge,
        "eaa_trust_selection": args.eaa_trust_selection,
        "eaa_policy": args.eaa_policy,
        "eaa_policy_config": (
            os.path.abspath(args.eaa_policy_config) if args.eaa_policy_config else None
        ),
        "eaa_traces": getattr(agent_class, "trace_log", []),
        "eaa": {
            "agent_id": "eaa",
            "pipeline": "evidence quality signals → policy → rank",
            "policy_mode": args.eaa_policy,
            "evidence_modules": ["H", "M", "S_prompt", "S_tool"],
        },
    }

    predictions_path = args.predictions_output or default_predictions_path(
        agent=args.agent,
        dataset=dataset,
        track=track,
        condition=args.condition,
        outputs_dir=args.predictions_dir,
    )
    save_predictions(
        path=predictions_path,
        metadata=run_metadata,
        simulation_outputs=simulator.simulation_outputs,
        groundtruth_data=simulator.groundtruth_data,
        task_indices=task_indices,
    )
    logger.info("Saved predictions to %s", predictions_path)

    metrics_path = args.output or default_metrics_path(
        agent=args.agent,
        dataset=dataset,
        track=track,
        condition=args.condition,
        results_dir=args.results_dir,
    )
    save_metrics_result(
        path=metrics_path,
        metadata={**run_metadata, "predictions_path": predictions_path},
        evaluation_results=evaluation_results,
        error_log=error_log,
        predictions_path=predictions_path,
    )
    logger.info("Saved metrics to %s", metrics_path)
    print(json.dumps(evaluation_results, indent=2))


if __name__ == "__main__":
    main()
