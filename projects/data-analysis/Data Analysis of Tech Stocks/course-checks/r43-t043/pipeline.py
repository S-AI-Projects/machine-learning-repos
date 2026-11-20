"""
Majority-class classification baseline for Data Analysis of Tech Stocks.
Parameters are fitted strictly on training data and evaluated on held-out test data.
"""

from typing import Any, Dict, List, Optional
from collections import Counter
import json


class MajorityClassBaseline:
    """
    Simplest honest baseline for classification:
    Predicts the majority class observed in the training split.
    """
    def __init__(self):
        self.majority_class: Optional[int] = None
        self.class_counts: Dict[int, int] = {}
        self.train_samples: int = 0

    def fit(self, train_records: List[Dict[str, Any]]) -> "MajorityClassBaseline":
        if not train_records:
            raise ValueError("Training records cannot be empty.")
        
        labels = [r["label"] for r in train_records]
        counts = Counter(labels)
        self.class_counts = dict(counts)
        self.train_samples = len(train_records)
        # Deterministic majority selection: highest frequency, tie-break by lower class value
        self.majority_class = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        return self

    def predict(self, records: List[Dict[str, Any]]) -> List[int]:
        if self.majority_class is None:
            raise RuntimeError("Baseline model has not been fitted yet.")
        return [self.majority_class for _ in records]

    def evaluate(self, test_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculates accuracy on held-out test records.
        """
        if not test_records:
            raise ValueError("Evaluation records cannot be empty.")
        
        y_true = [r["label"] for r in test_records]
        y_pred = self.predict(test_records)
        
        correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
        total = len(y_true)
        accuracy = correct / total if total > 0 else 0.0

        return {
            "baseline_strategy": "majority_class",
            "majority_class": self.majority_class,
            "train_sample_count": self.train_samples,
            "eval_sample_count": total,
            "accuracy": accuracy,
            "correct_predictions": correct,
            "notice": "Pipeline verification metric on synthetic data; no real-world accuracy claimed."
        }


def load_fixture(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
