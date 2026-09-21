#!/usr/bin/env python3
"""Generate oracle prediction runs needed for EAA threshold calibration."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--artifact_dir", required=True)
    parser.add_argument("--dataset", default="amazon")
    parser.add_argument("--track", type=int, default=2)
    parser.add_argument(
        "--prediction_prefix",
        default="eaa",
        help="Prefix for prediction/metrics filenames (use eaa_cal for external calibration pool).",
    )
    parser.add_argument("--predictions_dir", default=os.path.join(ROOT, "outputs"))
    parser.add_argument("--results_dir", default=os.path.join(ROOT, "results"))
    parser.add_argument("--llm_class", default="OllamaLLM")
    parser.add_argument("--ollama_model", default="qwen2.5:7b-instruct")
    parser.add_argument("--simulator_device", default="cpu")
    parser.add_argument("--max_tasks", type=int, default=None)
    return parser.parse_args()


def agent_tag(prefix: str, role: str) -> str:
    if prefix == "eaa":
        if role == "oracle_hm":
            return "eaa_oracle_hm"
        if role == "oracle_hmt":
            return "eaa_oracle_hmt"
        return "eaa"
    if role == "oracle_hm":
        return f"{prefix}_oracle_hm"
    if role == "oracle_hmt":
        return f"{prefix}_oracle_hmt"
    if role == "signals":
        return prefix
    return f"{prefix}_{role}"


def run_one(
    args: argparse.Namespace,
    *,
    role: str,
    modules: str | None,
    extra: list[str],
) -> None:
    tag = agent_tag(args.prediction_prefix, role)
    predictions_output = os.path.join(
        args.predictions_dir,
        f"{tag}_{args.dataset}_track{args.track}_{args.condition}_predictions.json",
    )
    cmd = [
        sys.executable,
        os.path.join(ROOT, "scripts", "run_condition_eval.py"),
        "--data_dir",
        args.data_dir,
        "--manifest",
        args.manifest,
        "--condition",
        args.condition,
        "--agent",
        "eaa",
        "--artifact_dir",
        args.artifact_dir,
        "--llm_class",
        args.llm_class,
        "--ollama_model",
        args.ollama_model,
        "--simulator_device",
        args.simulator_device,
        "--output",
        os.path.join(args.results_dir, f"{tag}_{args.dataset}_track{args.track}_{args.condition}.json"),
        "--predictions_output",
        predictions_output,
        "--predictions_dir",
        args.predictions_dir,
        *extra,
    ]
    if modules is not None:
        cmd.extend(["--eaa_force_modules", modules])
    if args.max_tasks is not None:
        cmd.extend(["--max_tasks", str(args.max_tasks)])
    print("Running:", " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([ROOT, os.path.join(ROOT, "scripts"), env.get("PYTHONPATH", "")])
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    args = parse_args()
    os.makedirs(args.predictions_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    run_one(args, role="oracle_hm", modules="H,M", extra=[])
    run_one(args, role="oracle_hmt", modules="H,M,S_tool", extra=[])
    run_one(args, role="signals", modules=None, extra=["--eaa_policy", "quality"])


if __name__ == "__main__":
    main()
