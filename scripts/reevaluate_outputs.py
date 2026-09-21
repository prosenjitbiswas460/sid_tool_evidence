#!/usr/bin/env python3
"""Recompute metrics from saved agent predictions without rerunning the LLM."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from websocietysimulator.evaluation_io import (
    evaluate_predictions_file,
    save_metrics_result,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("reevaluate_outputs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to a *_predictions.json file saved by run_condition_eval.py",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path for recomputed metrics JSON (default: results/<same basename without _predictions>)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="Device for simulation-track metrics only",
    )
    return parser.parse_args()


def default_metrics_output(predictions_path: str) -> str:
    base = os.path.basename(predictions_path)
    if base.endswith("_predictions.json"):
        base = base[: -len("_predictions.json")] + ".json"
    else:
        base = os.path.splitext(base)[0] + "_metrics.json"
    return os.path.join(ROOT, "results", base)


def main() -> None:
    args = parse_args()
    predictions_path = os.path.abspath(args.predictions)
    evaluation_results, error_log, metadata = evaluate_predictions_file(
        predictions_path,
        device=args.device,
    )

    output_path = args.output or default_metrics_output(predictions_path)
    save_metrics_result(
        path=output_path,
        metadata=metadata,
        evaluation_results=evaluation_results,
        error_log=error_log,
        predictions_path=predictions_path,
    )

    logger.info("Recomputed metrics from %s", predictions_path)
    logger.info("Saved metrics to %s", output_path)
    print(json.dumps(evaluation_results, indent=2))


if __name__ == "__main__":
    main()
