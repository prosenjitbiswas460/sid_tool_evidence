"""Save and reload agent predictions for offline metric recomputation."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from .tools.simple_evaluation_tool import (
    RecommendationEvaluator,
    SimulationEvaluator,
    recommendation_metrics_to_dict,
)


def default_predictions_path(
    agent: str,
    dataset: str,
    track: int,
    condition: str,
    outputs_dir: str,
) -> str:
    filename = f"{agent}_{dataset}_track{track}_{condition}_predictions.json"
    return os.path.join(outputs_dir, filename)


def default_metrics_path(
    agent: str,
    dataset: str,
    track: int,
    condition: str,
    results_dir: str,
) -> str:
    filename = f"{agent}_{dataset}_track{track}_{condition}.json"
    return os.path.join(results_dir, filename)


def build_prediction_records(
    simulation_outputs: List[Optional[Dict[str, Any]]],
    groundtruth_data: List[Dict[str, Any]],
    task_indices: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for index, (output, groundtruth) in enumerate(
        zip(simulation_outputs, groundtruth_data)
    ):
        record: Dict[str, Any] = {
            "run_index": index,
            "task_index": task_indices[index] if task_indices else index,
            "groundtruth": groundtruth,
        }
        if output is None:
            record["task"] = None
            record["output"] = None
            record["error"] = "missing output"
        elif "error" in output:
            record["task"] = output.get("task")
            record["output"] = None
            record["error"] = output["error"]
        else:
            record["task"] = output.get("task")
            record["output"] = output.get("output")
            record["error"] = None
        records.append(record)
    return records


def save_predictions(
    path: str,
    metadata: Dict[str, Any],
    simulation_outputs: List[Optional[Dict[str, Any]]],
    groundtruth_data: List[Dict[str, Any]],
    task_indices: Optional[List[int]] = None,
) -> str:
    payload = {
        "metadata": metadata,
        "records": build_prediction_records(
            simulation_outputs=simulation_outputs,
            groundtruth_data=groundtruth_data,
            task_indices=task_indices,
        ),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def load_predictions(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def evaluate_prediction_records(
    records: List[Dict[str, Any]],
    track: int,
    device: str = "auto",
) -> Tuple[Dict[str, Any], List[str]]:
    error_log = [
        record["error"]
        for record in records
        if record.get("error") and record["error"] != "missing output"
    ]

    if track == 2:
        ground_truth = [record["groundtruth"]["ground truth"] for record in records]
        predictions: List[List[str]] = []
        for record in records:
            if record.get("error") or record.get("output") is None:
                predictions.append([""])
            else:
                predictions.append(record["output"])

        metrics = RecommendationEvaluator().calculate_hr_at_n(
            ground_truth=ground_truth,
            predictions=predictions,
        )
        return {
            "type": "recommendation",
            "metrics": recommendation_metrics_to_dict(metrics),
        }, error_log

    if track == 1:
        ground_truth_data = [record["groundtruth"] for record in records]
        simulated_data = []
        for record in records:
            if record.get("error") or record.get("output") is None:
                simulated_data.append({"stars": 0, "review": ""})
            else:
                simulated_data.append(record["output"])

        metrics = SimulationEvaluator(device=device).calculate_metrics(
            simulated_data=simulated_data,
            ground_truth_data=ground_truth_data,
        )
        return {
            "type": "simulation",
            "metrics": metrics.__dict__,
        }, error_log

    raise ValueError(f"Unsupported track: {track}")


def evaluate_predictions_file(
    path: str,
    device: str = "auto",
) -> Tuple[Dict[str, Any], List[str], Dict[str, Any]]:
    payload = load_predictions(path)
    metadata = payload["metadata"]
    track = metadata["track"]
    evaluation_results, error_log = evaluate_prediction_records(
        records=payload["records"],
        track=track,
        device=device,
    )
    return evaluation_results, error_log, metadata


def save_metrics_result(
    path: str,
    metadata: Dict[str, Any],
    evaluation_results: Dict[str, Any],
    error_log: List[str],
    predictions_path: Optional[str] = None,
) -> str:
    payload = {
        "metadata": {
            **metadata,
            "predictions_path": os.path.abspath(predictions_path)
            if predictions_path
            else metadata.get("predictions_path"),
        },
        "evaluation_results": evaluation_results,
        "error_log": error_log,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path
