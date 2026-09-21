"""Unified evaluation metrics without sim/real task splits."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Union

import nltk
import numpy as np
import torch
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer
from transformers import pipeline

logger = logging.getLogger("websocietysimulator")


def ensure_nltk_data():
    """Ensure NLTK VADER lexicon is available."""
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        logging.warning("VADER lexicon not found, downloading...")
        nltk.download("vader_lexicon", quiet=True)


ensure_nltk_data()


@dataclass
class RecommendationMetrics:
    top_1_hit_rate: float
    top_3_hit_rate: float
    top_5_hit_rate: float
    top_10_hit_rate: float
    average_hit_rate: float
    total_scenarios: int
    top_1_hits: int
    top_3_hits: int
    top_5_hits: int
    top_10_hits: int
    hr_at_n: Dict[int, float]
    ndcg_at_n: Dict[int, float]
    recall_at_n: Dict[int, float]


@dataclass
class SimulationMetrics:
    preference_estimation: float
    review_generation: float
    overall_quality: float


class BaseEvaluator:
    """Base class for evaluation tools."""

    def __init__(self):
        self.metrics_history: List[Union[RecommendationMetrics, SimulationMetrics]] = []

    def save_metrics(self, metrics: Union[RecommendationMetrics, SimulationMetrics]):
        self.metrics_history.append(metrics)

    def get_metrics_history(self):
        return self.metrics_history


class RecommendationEvaluator(BaseEvaluator):
    """Evaluator for recommendation tasks."""

    def __init__(self):
        super().__init__()
        self.n_values = [1, 3, 5, 10]

    @staticmethod
    def _ndcg_at_k(prediction: List[str], ground_truth: str, k: int) -> float:
        top_k = prediction[:k]
        if ground_truth not in top_k:
            return 0.0
        rank = top_k.index(ground_truth) + 1
        dcg = 1.0 / np.log2(rank + 1)
        idcg = 1.0 / np.log2(2)
        return float(dcg / idcg)

    def calculate_hr_at_n(
        self,
        ground_truth: List[str],
        predictions: List[List[str]],
    ) -> RecommendationMetrics:
        """Calculate ranking metrics at k in {1, 3, 5, 10}."""
        total = len(ground_truth)
        hits = {n: 0 for n in self.n_values}
        ndcg_sums = {n: 0.0 for n in self.n_values}
        recall_sums = {n: 0.0 for n in self.n_values}

        for gt, pred in zip(ground_truth, predictions):
            for n in self.n_values:
                hit = gt in pred[:n]
                if hit:
                    hits[n] += 1
                ndcg_sums[n] += self._ndcg_at_k(pred, gt, n)
                # With one relevant item per task, recall@k equals hit rate@k.
                recall_sums[n] += 1.0 if hit else 0.0

        hr_at_n = {n: (hits[n] / total if total > 0 else 0.0) for n in self.n_values}
        ndcg_at_n = {n: (ndcg_sums[n] / total if total > 0 else 0.0) for n in self.n_values}
        recall_at_n = {n: (recall_sums[n] / total if total > 0 else 0.0) for n in self.n_values}

        top_1_hit_rate = hr_at_n[1]
        top_3_hit_rate = hr_at_n[3]
        top_5_hit_rate = hr_at_n[5]
        top_10_hit_rate = hr_at_n[10]
        average_hit_rate = (top_1_hit_rate + top_3_hit_rate + top_5_hit_rate) / 3

        metrics = RecommendationMetrics(
            top_1_hit_rate=top_1_hit_rate,
            top_3_hit_rate=top_3_hit_rate,
            top_5_hit_rate=top_5_hit_rate,
            top_10_hit_rate=top_10_hit_rate,
            average_hit_rate=average_hit_rate,
            total_scenarios=total,
            top_1_hits=hits[1],
            top_3_hits=hits[3],
            top_5_hits=hits[5],
            top_10_hits=hits[10],
            hr_at_n=hr_at_n,
            ndcg_at_n=ndcg_at_n,
            recall_at_n=recall_at_n,
        )
        self.save_metrics(metrics)
        return metrics


def recommendation_metrics_to_dict(metrics: RecommendationMetrics) -> Dict[str, Union[int, float, Dict[str, float]]]:
    """Serialize recommendation metrics with explicit @k fields for JSON output."""
    payload: Dict[str, Union[int, float, Dict[str, float]]] = {
        "top_1_hit_rate": metrics.top_1_hit_rate,
        "top_3_hit_rate": metrics.top_3_hit_rate,
        "top_5_hit_rate": metrics.top_5_hit_rate,
        "top_10_hit_rate": metrics.top_10_hit_rate,
        "average_hit_rate": metrics.average_hit_rate,
        "total_scenarios": metrics.total_scenarios,
        "top_1_hits": metrics.top_1_hits,
        "top_3_hits": metrics.top_3_hits,
        "top_5_hits": metrics.top_5_hits,
        "top_10_hits": metrics.top_10_hits,
        "hr_at_n": {str(k): v for k, v in metrics.hr_at_n.items()},
        "ndcg_at_n": {str(k): v for k, v in metrics.ndcg_at_n.items()},
        "recall_at_n": {str(k): v for k, v in metrics.recall_at_n.items()},
    }
    for k, value in metrics.hr_at_n.items():
        payload[f"hr@{k}"] = value
    for k, value in metrics.ndcg_at_n.items():
        payload[f"ndcg@{k}"] = value
    for k, value in metrics.recall_at_n.items():
        payload[f"recall@{k}"] = value
    return payload


class SimulationEvaluator(BaseEvaluator):
    """Evaluator for user-behavior simulation tasks."""

    def __init__(self, device: str = "auto"):
        super().__init__()
        self.device = self._get_device(device)

        pipeline_device = self.device
        st_device = "cuda" if self.device == 0 else "cpu"

        self.sia = SentimentIntensityAnalyzer()
        self.emotion_classifier = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-emotion",
            top_k=5,
            device=pipeline_device,
        )
        self.topic_model = SentenceTransformer(
            "paraphrase-MiniLM-L6-v2",
            device=st_device,
        )

    def _get_device(self, device: str) -> int:
        if device == "gpu":
            if torch.cuda.is_available():
                return 0
            logging.warning("GPU is not available, falling back to CPU")
            return -1
        if device == "cpu":
            return -1
        if device == "auto":
            return 0 if torch.cuda.is_available() else -1
        raise ValueError("Device type must be 'cpu', 'gpu' or 'auto'")

    def calculate_metrics(
        self,
        simulated_data: List[Dict],
        ground_truth_data: List[Dict],
    ) -> SimulationMetrics:
        """Calculate simulation metrics against ground truth for all tasks."""
        if not simulated_data or not ground_truth_data:
            return SimulationMetrics(
                preference_estimation=0.0,
                review_generation=0.0,
                overall_quality=0.0,
            )

        simulated_stars = [item["stars"] for item in simulated_data]
        ground_truth_stars = [item["stars"] for item in ground_truth_data]

        star_error = 0.0
        for sim_star, gt_star in zip(simulated_stars, ground_truth_stars):
            sim_star = min(max(sim_star, 0), 5)
            star_error += abs(sim_star - gt_star) / 5

        star_error /= len(ground_truth_stars)
        preference_estimation = 1 - star_error

        simulated_reviews = [item["review"] for item in simulated_data]
        ground_truth_reviews = [item["review"] for item in ground_truth_data]
        review_details = self._calculate_review_metrics(
            simulated_reviews,
            ground_truth_reviews,
        )

        review_generation = 1 - (
            review_details["sentiment_error"] * 0.25
            + review_details["emotion_error"] * 0.25
            + review_details["topic_error"] * 0.5
        )
        overall_quality = (preference_estimation + review_generation) / 2

        metrics = SimulationMetrics(
            preference_estimation=preference_estimation,
            review_generation=review_generation,
            overall_quality=overall_quality,
        )
        self.save_metrics(metrics)
        return metrics

    def _calculate_review_metrics(
        self,
        simulated_reviews: List[str],
        ground_truth_reviews: List[str],
    ) -> Dict[str, float]:
        sentiment_error = []
        topic_error = []

        for simulated_review, ground_truth_review in zip(simulated_reviews, ground_truth_reviews):
            sentiment1 = self.sia.polarity_scores(simulated_review)["compound"]
            sentiment2 = self.sia.polarity_scores(ground_truth_review)["compound"]
            sentiment_error.append(abs(sentiment1 - sentiment2) / 2)

            embeddings = self.topic_model.encode([simulated_review, ground_truth_review])
            topic_error.append(distance.cosine(embeddings[0], embeddings[1]) / 2)

        truncated_simulated = [
            review[:300] if len(review) > 300 else review for review in simulated_reviews
        ]
        truncated_ground_truth = [
            review[:300] if len(review) > 300 else review for review in ground_truth_reviews
        ]
        simulated_emotions = self.emotion_classifier(truncated_simulated)
        ground_truth_emotions = self.emotion_classifier(truncated_ground_truth)

        emotion_error = [
            self._calculate_emotion_error(sim_emotion, gt_emotion)
            for sim_emotion, gt_emotion in zip(simulated_emotions, ground_truth_emotions)
        ]

        return {
            "sentiment_error": float(np.mean(sentiment_error)) if sentiment_error else 0.0,
            "emotion_error": float(np.mean(emotion_error)) if emotion_error else 0.0,
            "topic_error": float(np.mean(topic_error)) if topic_error else 0.0,
        }

    def _calculate_emotion_error(
        self,
        emotions1: List[Dict],
        emotions2: List[Dict],
    ) -> float:
        emotion_dict1 = {e["label"]: e["score"] for e in emotions1}
        emotion_dict2 = {e["label"]: e["score"] for e in emotions2}
        all_emotions = set(emotion_dict1.keys()) | set(emotion_dict2.keys())
        vec1 = np.array([emotion_dict1.get(e, 0) for e in all_emotions])
        vec2 = np.array([emotion_dict2.get(e, 0) for e in all_emotions])
        return float(np.mean(np.abs(vec1 - vec2)))
