"""In-process API metrics and a simple transparent PSI drift statistic."""

from __future__ import annotations

import math
import threading

import numpy as np


class RequestMetrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.requests = 0
        self.errors = 0
        self.validation_errors = 0
        self.latencies_ms: list[float] = []
        self.fraud_scores: list[float] = []
        self.risk_counts: dict[str, int] = {"low": 0, "medium": 0, "high": 0}

    def record(self, elapsed_ms: float, error: bool = False, validation_error: bool = False):
        with self._lock:
            self.requests += 1
            self.errors += int(error)
            self.validation_errors += int(validation_error)
            self.latencies_ms.append(float(elapsed_ms))
            self.latencies_ms = self.latencies_ms[-10_000:]

    def record_prediction(self, score: float, risk: str):
        with self._lock:
            self.fraud_scores.append(float(score))
            self.fraud_scores = self.fraud_scores[-10_000:]
            self.risk_counts[risk] = self.risk_counts.get(risk, 0) + 1

    def snapshot(self):
        with self._lock:
            data = list(self.latencies_ms)
            requests, errors = self.requests, self.errors
            risk_counts = dict(self.risk_counts)
            prediction_scores = list(self.fraud_scores)
            validation_errors = self.validation_errors
        return {
            "request_count": requests, "error_count": errors,
            "error_rate": errors / requests if requests else None,
            "validation_error_count": validation_errors,
            "latency_ms_p50": float(np.percentile(data, 50)) if data else None,
            "latency_ms_p95": float(np.percentile(data, 95)) if data else None,
            "latency_samples": len(data),
            "prediction_count": len(prediction_scores),
            "risk_category_counts": risk_counts,
            "mean_fraud_score": float(np.mean(prediction_scores)) if prediction_scores else None,
            "fraud_score_histogram": np.histogram(prediction_scores, bins=np.linspace(0, 1, 11))[0].tolist() if prediction_scores else [],
        }


def population_stability_index(reference, recent, bins: int = 10) -> float | None:
    """PSI with reference-derived bins; None when either sample has no finite values."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(recent, dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if not len(ref) or not len(cur):
        return None
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        edges = np.array([ref.min() - 1e-9, ref.max() + 1e-9])
    edges[0] = -math.inf
    edges[-1] = math.inf
    expected = np.histogram(ref, bins=edges)[0] / len(ref)
    actual = np.histogram(cur, bins=edges)[0] / len(cur)
    expected = np.clip(expected, 1e-6, None)
    actual = np.clip(actual, 1e-6, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))
