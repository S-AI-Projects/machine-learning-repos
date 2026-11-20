"""
Evaluation metric verifier for World Population Growth Analysis.
Explicitly defines precision, recall, MAE, and zero-denominator handling.
"""

from typing import Any, Dict, List, Sequence
import json


def calculate_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    zero_division: float = 0.0
) -> Dict[str, float]:
    """
    Calculates binary classification metrics with explicit zero-denominator behavior:
    - precision = tp / (tp + fp), defaults to zero_division if (tp + fp) == 0
    - recall = tp / (tp + fn), defaults to zero_division if (tp + fn) == 0
    - accuracy = (tp + tn) / total
    - f1 = 2 * precision * recall / (precision + recall)
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_true)}, len(y_pred)={len(y_pred)}")
    
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    total = len(y_true)

    accuracy = (tp + tn) / total if total > 0 else zero_division
    precision = tp / (tp + fp) if (tp + fp) > 0 else zero_division
    recall = tp / (tp + fn) if (tp + fn) > 0 else zero_division
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else zero_division

    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def calculate_mae(
    y_true: Sequence[float],
    y_pred: Sequence[float]
) -> float:
    """
    Calculates Mean Absolute Error for regression cross-checks:
    MAE = (1/N) * sum(|y_true - y_pred|)
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Length mismatch between y_true and y_pred.")
    if len(y_true) == 0:
        return 0.0
    return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / len(y_true)


def load_fixture(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
