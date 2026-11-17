"""
Evaluation metric verifier for Handwritten Digit Recognition image fixture.
Explicitly defines precision, recall, MAE, and zero-denominator handling.
"""

from typing import Any, Dict, List, Sequence


def calculate_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    zero_division: float = 0.0
) -> Dict[str, float]:
    """
    Calculates binary classification metrics with explicit zero-denominator behavior.
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


def classify_diagonal_image(image_matrix: List[List[int]]) -> int:
    """
    Deterministic rule-based classifier for 4x4 image fixture:
    Returns 1 if the main diagonal has non-zero intensity, else 0.
    """
    diagonal_sum = sum(image_matrix[i][i] for i in range(min(len(image_matrix), len(image_matrix[0]))))
    return 1 if diagonal_sum > 0 else 0
