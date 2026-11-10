import numpy as np
from sentinel_ai.models.evaluation import evaluate, select_threshold
from sentinel_ai.monitoring.metrics import RequestMetrics, population_stability_index


def test_threshold_uses_validation_precision_constraint():
    y = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    threshold = select_threshold(y, scores, min_precision=0.5)
    assert threshold <= 0.8
    assert evaluate(y, scores, threshold)["recall"] == 1.0


def test_evaluation_reports_confusion_and_false_positive_rate():
    result = evaluate([0, 0, 1, 1], [0.8, 0.1, 0.7, 0.2], 0.5)
    assert result["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert result["false_positives_per_1000"] == 250


def test_psi_empty_and_shifted_distributions():
    assert population_stability_index([], [1]) is None
    assert population_stability_index(np.arange(100), np.arange(100) + 100) > 0.2


def test_operational_metrics_empty_then_recorded():
    metrics = RequestMetrics()
    assert metrics.snapshot()["latency_ms_p95"] is None
    metrics.record(12, error=True)
    metrics.record_prediction(0.91, "high")
    result = metrics.snapshot()
    assert result["request_count"] == 1 and result["error_count"] == 1
    assert result["risk_category_counts"]["high"] == 1
