"""Classification metrics and validation-only threshold selection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)


def select_threshold(y_true, probabilities, min_precision: float = 0.50) -> float:
    """Choose max-recall threshold among validation points meeting precision floor."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    candidates = [
        (recall[i], thresholds[i]) for i in range(len(thresholds))
        if precision[i] >= min_precision
    ]
    if candidates:
        return float(max(candidates, key=lambda item: (item[0], item[1]))[1])
    # If the constraint is unsupported by validation, use the F1-optimal point.
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))]) if len(thresholds) else 0.5


def evaluate(y_true, probabilities, threshold: float) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    labels = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, labels, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, labels, zero_division=0)),
        "recall": float(recall_score(y_true, labels, zero_division=0)),
        "f1": float(f1_score(y_true, labels, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) > 1 else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "false_positives_per_1000": float(fp / max(len(y_true), 1) * 1000),
    }


def curve_points(y_true, probabilities) -> dict:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    return {"precision": precision.tolist(), "recall": recall.tolist(),
            "thresholds": thresholds.tolist()}
