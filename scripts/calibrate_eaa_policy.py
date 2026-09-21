#!/usr/bin/env python3
"""
Calibrate EAA v4 policy thresholds.

Modes:
  split (default): 70/30 split on the provided prediction set (legacy benchmark tuning).
  external: tune on all tasks from a disjoint calibration pool; evaluate on full benchmark.

External calibration workflow:
  1. scripts/build_calibration_manifest.py
  2. scripts/run_eaa_calibration_oracles.py --manifest <calibration> --prediction_prefix eaa_cal
  3. This script with --calibration_mode external
  4. scripts/run_condition_eval.py on benchmark manifest with --eaa_policy_config
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import sys
from typing import Any, Dict, Iterable, List, Sequence, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXAMPLE = os.path.join(ROOT, "example")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if EXAMPLE not in sys.path:
    sys.path.insert(0, EXAMPLE)

from eaa_evidence_policy import (  # noqa: E402
    EvidencePolicyConfig,
    save_policy_config,
    select_modules_quality,
    signals_from_dict,
)
from websocietysimulator.evaluation_io import load_predictions  # noqa: E402

PolicyKey = str  # "hm" | "hmt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions_dir", default=os.path.join(ROOT, "outputs"))
    parser.add_argument("--dataset", default="amazon")
    parser.add_argument("--track", type=int, default=2)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--eaa_agent", default="eaa")
    parser.add_argument("--oracle_hm_agent", default="eaa_oracle_hm")
    parser.add_argument("--oracle_hmt_agent", default="eaa_oracle_hmt")
    parser.add_argument(
        "--calibration_mode",
        choices=["split", "external"],
        default="split",
        help="split=70/30 on prediction tasks; external=tune on full calibration pool",
    )
    parser.add_argument(
        "--calibration_manifest",
        default=None,
        help="Calibration pool manifest (required for external mode metadata).",
    )
    parser.add_argument(
        "--benchmark_manifest",
        default=None,
        help="Official benchmark manifest used for final evaluation reporting.",
    )
    parser.add_argument(
        "--validation_ratio",
        type=float,
        default=0.7,
        help="Fraction of tasks used for threshold tuning (default 70%% validation).",
    )
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument(
        "--history_reviews_grid",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5, 6, 7, 8],
    )
    parser.add_argument(
        "--metadata_grid",
        nargs="+",
        type=float,
        default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55],
    )
    parser.add_argument(
        "--sid_margin_grid",
        nargs="+",
        type=float,
        default=[0.01, 0.02, 0.05, 0.08, 0.10],
    )
    parser.add_argument(
        "--sid_top1_grid",
        nargs="+",
        type=float,
        default=[0.10, 0.15, 0.20, 0.25],
    )
    parser.add_argument("--history_chars", type=int, default=2000)
    parser.add_argument(
        "--sensitivity_reviews_window",
        type=int,
        default=2,
        help="± window around optimal history threshold for sensitivity table.",
    )
    parser.add_argument("--bootstrap_samples", type=int, default=200)
    parser.add_argument(
        "--output",
        default=os.path.join(ROOT, "configs", "eaa_policy_calibrated.json"),
    )
    return parser.parse_args()


def ndcg_at_k(prediction: Sequence[str], ground_truth: str, k: int = 5) -> float:
    top_k = list(prediction)[:k]
    if ground_truth not in top_k:
        return 0.0
    rank = top_k.index(ground_truth) + 1
    return 1.0 / math.log2(rank + 1)


def per_task_ndcg(path: str, k: int = 5) -> List[float]:
    payload = load_predictions(path)
    scores: List[float] = []
    for record in payload["records"]:
        gt = record["groundtruth"]["ground truth"]
        if record.get("error") or record.get("output") is None:
            scores.append(0.0)
            continue
        scores.append(ndcg_at_k(record["output"], gt, k=k))
    return scores


def predictions_path(
    predictions_dir: str,
    agent: str,
    dataset: str,
    track: int,
    condition: str,
) -> str:
    return os.path.join(
        predictions_dir,
        f"{agent}_{dataset}_track{track}_{condition}_predictions.json",
    )


def resolve_predictions_path(
    predictions_dir: str,
    agent: str,
    dataset: str,
    track: int,
    condition: str,
    *,
    legacy_agents: Sequence[str] = (),
) -> str:
    """Resolve prediction file, with optional legacy agent id fallbacks."""
    candidates = [agent, *legacy_agents]
    for candidate in candidates:
        path = predictions_path(predictions_dir, candidate, dataset, track, condition)
        if os.path.isfile(path):
            return path
    return predictions_path(predictions_dir, agent, dataset, track, condition)


def load_task_rows(
    *,
    predictions_dir: str,
    dataset: str,
    track: int,
    condition: str,
    eaa_agent: str,
    oracle_hm_agent: str,
    oracle_hmt_agent: str,
) -> List[Dict[str, Any]]:
    signal_path = resolve_predictions_path(
        predictions_dir,
        eaa_agent,
        dataset,
        track,
        condition,
        legacy_agents=(f"{eaa_agent}_signals",),
    )
    hm_path = predictions_path(predictions_dir, oracle_hm_agent, dataset, track, condition)
    hmt_path = predictions_path(predictions_dir, oracle_hmt_agent, dataset, track, condition)

    for path in (signal_path, hm_path, hmt_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing predictions file: {path}\n"
                "Run: python scripts/run_eaa_calibration_oracles.py ..."
            )

    signal_payload = load_predictions(signal_path)
    traces = signal_payload.get("metadata", {}).get("eaa_traces") or []
    hm_scores = per_task_ndcg(hm_path)
    hmt_scores = per_task_ndcg(hmt_path)
    records = signal_payload["records"]
    count = min(len(traces), len(hm_scores), len(hmt_scores), len(records))
    if count == 0:
        raise ValueError(f"No aligned tasks for condition={condition}")

    rows: List[Dict[str, Any]] = []
    for index in range(count):
        trace = traces[index]
        signal_dict = trace.get("evidence_signals") or trace.get("signals")
        if not signal_dict:
            raise ValueError(f"Task {index} missing evidence_signals in eaa_traces.")
        ndcg_hm = hm_scores[index]
        ndcg_hmt = hmt_scores[index]
        oracle_key: PolicyKey = "hmt" if ndcg_hmt > ndcg_hm else "hm"
        rows.append(
            {
                "task_index": records[index].get("task_index", index),
                "run_index": index,
                "signals": signals_from_dict(signal_dict),
                "ndcg_hm": ndcg_hm,
                "ndcg_hmt": ndcg_hmt,
                "oracle_ndcg@5": max(ndcg_hm, ndcg_hmt),
                "oracle_key": oracle_key,
            }
        )
    return rows


def split_validation_heldout(
    n: int,
    validation_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1.")
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    validation_size = max(1, int(round(n * validation_ratio)))
    if validation_size >= n:
        validation_size = n - 1
    validation_idx = sorted(indices[:validation_size])
    heldout_idx = sorted(indices[validation_size:])
    if not heldout_idx:
        heldout_idx = [validation_idx.pop()]
    return validation_idx, heldout_idx


def policy_key(row: Dict[str, Any], config: EvidencePolicyConfig) -> PolicyKey:
    modules, _ = select_modules_quality(row["signals"], config)
    return "hmt" if "S_tool" in modules else "hm"


def counterfactual_oracle_ndcg(
    rows: Iterable[Dict[str, Any]],
    config: EvidencePolicyConfig,
) -> float:
    """
    Validation tuning only: NDCG@5 if policy-selected config used oracle ranking.
    Not used for held-out policy evaluation.
    """
    scores: List[float] = []
    for row in rows:
        key = policy_key(row, config)
        scores.append(row["ndcg_hmt"] if key == "hmt" else row["ndcg_hm"])
    return sum(scores) / len(scores) if scores else 0.0


def evidence_selection_accuracy(
    rows: Iterable[Dict[str, Any]],
    config: EvidencePolicyConfig,
) -> float:
    rows = list(rows)
    if not rows:
        return 0.0
    correct = sum(policy_key(row, config) == row["oracle_key"] for row in rows)
    return correct / len(rows)


def evidence_configuration_usage(
    rows: Sequence[Dict[str, Any]],
    config: EvidencePolicyConfig,
) -> Dict[str, Any]:
    if not rows:
        return {}
    hm = sum(policy_key(row, config) == "hm" for row in rows)
    total = len(rows)
    return {
        "H,M": {"count": hm, "percent": round(100.0 * hm / total, 1)},
        "H,M,S_tool": {
            "count": total - hm,
            "percent": round(100.0 * (total - hm) / total, 1),
        },
    }


def grid_search_thresholds(
    validation_rows: Sequence[Dict[str, Any]],
    *,
    history_reviews_grid: Sequence[int],
    metadata_grid: Sequence[float],
    sid_margin_grid: Sequence[float],
    sid_top1_grid: Sequence[float],
    history_chars: int,
) -> Tuple[EvidencePolicyConfig, float, float, int]:
    best_config: EvidencePolicyConfig | None = None
    best_accuracy = -1.0
    best_counterfactual_ndcg = -1.0
    tie_count = 0

    for reviews, metadata, margin, top1 in itertools.product(
        history_reviews_grid,
        metadata_grid,
        sid_margin_grid,
        sid_top1_grid,
    ):
        config = EvidencePolicyConfig(
            history_dense_reviews=reviews,
            sparse_history_reviews=reviews,
            history_dense_chars=history_chars,
            metadata_rich_threshold=metadata,
            sid_margin_threshold=margin,
            sid_top1_threshold=top1,
            policy_version="v4_calibrated",
        )
        accuracy = evidence_selection_accuracy(validation_rows, config)
        counterfactual_ndcg = counterfactual_oracle_ndcg(validation_rows, config)
        if accuracy > best_accuracy or (
            math.isclose(accuracy, best_accuracy)
            and counterfactual_ndcg > best_counterfactual_ndcg
        ):
            best_accuracy = accuracy
            best_counterfactual_ndcg = counterfactual_ndcg
            best_config = config
            tie_count = 1
        elif best_config and math.isclose(accuracy, best_accuracy) and math.isclose(
            counterfactual_ndcg, best_counterfactual_ndcg
        ):
            tie_count += 1

    if best_config is None:
        raise RuntimeError("Grid search produced no configuration.")
    return best_config, best_accuracy, best_counterfactual_ndcg, tie_count


def sensitivity_sweep(
    rows: Sequence[Dict[str, Any]],
    base_config: EvidencePolicyConfig,
    *,
    reviews_window: int,
    reviews_grid: Sequence[int],
) -> Dict[str, Any]:
    """Validation sensitivity: ESA + counterfactual oracle NDCG (tuning analysis only)."""
    reviews_values = sorted(
        {
            value
            for value in reviews_grid
            if abs(value - base_config.history_dense_reviews) <= reviews_window
        }
    )
    reviews_curve = []
    for reviews in reviews_values:
        config = EvidencePolicyConfig(
            history_dense_reviews=reviews,
            sparse_history_reviews=reviews,
            history_dense_chars=base_config.history_dense_chars,
            metadata_rich_threshold=base_config.metadata_rich_threshold,
            sid_margin_threshold=base_config.sid_margin_threshold,
            sid_top1_threshold=base_config.sid_top1_threshold,
            policy_version="v4_calibrated",
        )
        reviews_curve.append(
            {
                "history_dense_reviews": reviews,
                "mean_ndcg@5": counterfactual_oracle_ndcg(rows, config),
                "evidence_selection_accuracy": evidence_selection_accuracy(rows, config),
            }
        )

    metadata_curve = []
    for metadata in sorted(
        {
            base_config.metadata_rich_threshold - 0.05,
            base_config.metadata_rich_threshold,
            base_config.metadata_rich_threshold + 0.05,
        }
    ):
        if metadata <= 0:
            continue
        config = EvidencePolicyConfig(
            history_dense_reviews=base_config.history_dense_reviews,
            sparse_history_reviews=base_config.sparse_history_reviews,
            history_dense_chars=base_config.history_dense_chars,
            metadata_rich_threshold=metadata,
            sid_margin_threshold=base_config.sid_margin_threshold,
            sid_top1_threshold=base_config.sid_top1_threshold,
            policy_version="v4_calibrated",
        )
        metadata_curve.append(
            {
                "metadata_rich_threshold": metadata,
                "mean_ndcg@5": counterfactual_oracle_ndcg(rows, config),
                "evidence_selection_accuracy": evidence_selection_accuracy(rows, config),
            }
        )

    return {
        "history_dense_reviews": reviews_curve,
        "metadata_rich_threshold": metadata_curve,
        "note": (
            "Validation-only counterfactual NDCG from oracle forced-module runs; "
            "used for threshold sensitivity during tuning, not held-out policy scores."
        ),
    }


def bootstrap_threshold_ci(
    validation_rows: Sequence[Dict[str, Any]],
    *,
    base_config: EvidencePolicyConfig,
    history_reviews_grid: Sequence[int],
    metadata_grid: Sequence[float],
    sid_margin_grid: Sequence[float],
    sid_top1_grid: Sequence[float],
    history_chars: int,
    n_samples: int,
    seed: int,
) -> Dict[str, Any]:
    if not validation_rows:
        return {}
    rng = random.Random(seed)
    n = len(validation_rows)
    review_counts: List[int] = []
    metadata_values: List[float] = []
    for _ in range(n_samples):
        sample = [validation_rows[rng.randrange(n)] for _ in range(n)]
        best, _, _, _ = grid_search_thresholds(
            sample,
            history_reviews_grid=history_reviews_grid,
            metadata_grid=metadata_grid,
            sid_margin_grid=sid_margin_grid,
            sid_top1_grid=sid_top1_grid,
            history_chars=history_chars,
        )
        review_counts.append(best.history_dense_reviews)
        metadata_values.append(best.metadata_rich_threshold)

    def percentile(values: List[float], q: float) -> float:
        ordered = sorted(values)
        idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
        return ordered[idx]

    return {
        "bootstrap_samples": n_samples,
        "history_dense_reviews": {
            "optimal": base_config.history_dense_reviews,
            "ci_95_low": percentile([float(v) for v in review_counts], 0.025),
            "ci_95_high": percentile([float(v) for v in review_counts], 0.975),
        },
        "metadata_rich_threshold": {
            "optimal": base_config.metadata_rich_threshold,
            "ci_95_low": percentile(metadata_values, 0.025),
            "ci_95_high": percentile(metadata_values, 0.975),
        },
    }


def main() -> None:
    args = parse_args()
    if args.calibration_mode == "external":
        if not args.calibration_manifest:
            raise ValueError("--calibration_manifest is required for external mode.")
        if args.oracle_hm_agent == "eaa_oracle_hm" and args.eaa_agent == "eaa":
            args.oracle_hm_agent = "eaa_cal_oracle_hm"
            args.oracle_hmt_agent = "eaa_cal_oracle_hmt"
            args.eaa_agent = "eaa_cal"

    rows = load_task_rows(
        predictions_dir=args.predictions_dir,
        dataset=args.dataset,
        track=args.track,
        condition=args.condition,
        eaa_agent=args.eaa_agent,
        oracle_hm_agent=args.oracle_hm_agent,
        oracle_hmt_agent=args.oracle_hmt_agent,
    )

    if args.calibration_mode == "external":
        validation_idx = list(range(len(rows)))
        heldout_idx: List[int] = []
        validation_rows = rows
        heldout_rows: List[Dict[str, Any]] = []
        validation_task_indices = [row["task_index"] for row in rows]
        heldout_task_indices: List[int] = []
    else:
        validation_idx, heldout_idx = split_validation_heldout(
            len(rows),
            args.validation_ratio,
            args.split_seed,
        )
        validation_rows = [rows[i] for i in validation_idx]
        heldout_rows = [rows[i] for i in heldout_idx]
        validation_task_indices = [rows[i]["task_index"] for i in validation_idx]
        heldout_task_indices = [rows[i]["task_index"] for i in heldout_idx]

    best_config, val_esa, val_counterfactual_ndcg, ties = grid_search_thresholds(
        validation_rows,
        history_reviews_grid=args.history_reviews_grid,
        metadata_grid=args.metadata_grid,
        sid_margin_grid=args.sid_margin_grid,
        sid_top1_grid=args.sid_top1_grid,
        history_chars=args.history_chars,
    )

    sensitivity_validation = sensitivity_sweep(
        validation_rows,
        best_config,
        reviews_window=args.sensitivity_reviews_window,
        reviews_grid=args.history_reviews_grid,
    )
    threshold_ci = bootstrap_threshold_ci(
        validation_rows,
        base_config=best_config,
        history_reviews_grid=args.history_reviews_grid,
        metadata_grid=args.metadata_grid,
        sid_margin_grid=args.sid_margin_grid,
        sid_top1_grid=args.sid_top1_grid,
        history_chars=args.history_chars,
        n_samples=args.bootstrap_samples,
        seed=args.split_seed + 1,
    )

    calibration_meta = {
        "condition": args.condition,
        "dataset": args.dataset,
        "track": args.track,
        "calibration_mode": args.calibration_mode,
        "calibration_manifest": (
            os.path.abspath(args.calibration_manifest) if args.calibration_manifest else None
        ),
        "benchmark_manifest": (
            os.path.abspath(args.benchmark_manifest) if args.benchmark_manifest else None
        ),
        "split_seed": args.split_seed,
        "validation_ratio": 1.0 if args.calibration_mode == "external" else args.validation_ratio,
        "heldout_ratio": 0.0 if args.calibration_mode == "external" else 1.0 - args.validation_ratio,
        "validation_task_indices": validation_task_indices,
        "heldout_task_indices": heldout_task_indices,
        "validation_task_count": len(validation_rows),
        "heldout_task_count": len(heldout_rows),
        "policy_scope": "adaptive selection between two evidence configurations (H,M vs H,M,S_tool)",
        "evidence_selection_accuracy_definition": (
            "ESA = #{tasks where policy chooses oracle evidence configuration} / #tasks"
        ),
        "tuning_objective": (
            "maximize evidence_selection_accuracy on validation "
            "(tie-break: validation counterfactual oracle NDCG@5)"
        ),
        "validation_evidence_selection_accuracy": val_esa,
        "validation_counterfactual_mean_ndcg@5": val_counterfactual_ndcg,
        "validation_evidence_configuration_usage": evidence_configuration_usage(
            validation_rows, best_config
        ),
        "heldout_expected_evidence_configuration_usage": evidence_configuration_usage(
            heldout_rows, best_config
        ) if heldout_rows else {},
        "optimal_threshold_bootstrap_ci": threshold_ci,
        "sensitivity_validation": sensitivity_validation,
        "search_space": {
            "history_reviews_grid": args.history_reviews_grid,
            "metadata_grid": args.metadata_grid,
            "sid_margin_grid": args.sid_margin_grid,
            "sid_top1_grid": args.sid_top1_grid,
            "history_dense_chars_fixed": args.history_chars,
        },
        "oracle_agents": {
            "hm": args.oracle_hm_agent,
            "hmt": args.oracle_hmt_agent,
            "signals": args.eaa_agent,
        },
        "validation_ties_at_optimum": ties,
        "benchmark_evaluation_protocol": (
            "Execute calibrated policy on the full official benchmark manifest "
            "(same task count as A3/plum). Thresholds tuned only on disjoint calibration pool."
        ),
        "methodology_note": (
            "Thresholds tuned on a calibration pool sampled from processed reviews, "
            "disjoint from official benchmark (user, item) pairs. Final agents are "
            "evaluated on the full benchmark with frozen thresholds."
            if args.calibration_mode == "external"
            else (
                "We randomly partition benchmark tasks into 70% validation and 30% "
                "held-out evaluation (seed 42). Oracle forced-module runs are used "
                "only on validation to label best evidence configuration and tune "
                "thresholds. Policy structure is fixed; only hyperparameters are tuned."
            )
        ),
    }

    if args.calibration_mode == "external":
        calibration_meta.pop("heldout_evaluation_protocol", None)
    else:
        calibration_meta["heldout_evaluation_protocol"] = (
            "Execute calibrated policy on held-out tasks: compute evidence signals, "
            "apply frozen thresholds, choose H,M or H,M,S_tool, run one ranking "
            "configuration, evaluate NDCG@5. No oracle at runtime. "
            "Post-hoc regret: scripts/analyze_eaa_policy_regret.py"
        )

    save_policy_config(args.output, best_config, calibration=calibration_meta)

    summary = {
        "output": os.path.abspath(args.output),
        "calibration_mode": args.calibration_mode,
        "optimal_thresholds": best_config.to_dict(),
        "validation_evidence_selection_accuracy": val_esa,
        "validation_counterfactual_mean_ndcg@5": val_counterfactual_ndcg,
        "validation_evidence_configuration_usage": evidence_configuration_usage(
            validation_rows, best_config
        ),
        "sensitivity_validation": sensitivity_validation,
        "optimal_threshold_bootstrap_ci": threshold_ci,
        "next_steps": (
            [
                "Evaluate on benchmark: run_condition_eval.py --manifest <benchmark> "
                "--eaa_policy_config <this file> --eval_split all",
                "Post-hoc regret: analyze_eaa_policy_regret.py --eval_scope benchmark",
            ]
            if args.calibration_mode == "external"
            else [
                "Run held-out policy: run_condition_eval.py --calibration_config ... --eval_split heldout",
                "Post-hoc regret: analyze_eaa_policy_regret.py --calibration_config ...",
            ]
        ),
    }
    if args.calibration_mode != "external":
        summary["heldout_task_indices"] = heldout_task_indices
        summary["heldout_expected_evidence_configuration_usage"] = evidence_configuration_usage(
            heldout_rows, best_config
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
