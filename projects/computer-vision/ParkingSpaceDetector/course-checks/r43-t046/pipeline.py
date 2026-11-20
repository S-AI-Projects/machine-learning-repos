"""
Bounded CPU smoke run pipeline for ParkingSpaceDetector.
Operates fully offline on CPU without GPU, remote downloads, or credentials.
Trains for at most 1 epoch / <= 20 update steps and records loss and step counts.
"""

import math
from typing import Any, Dict, List, Tuple
import json


class BoundedCPUSmokeClassifier:
    """
    Lightweight, deterministic binary logistic regression model for CPU smoke testing.
    Asserts interface correctness, finite loss, and bounded step counts.
    """
    def __init__(self, learning_rate: float = 0.05, max_steps: int = 20, seed: int = 42):
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.seed = seed
        self.weights = [0.0, 0.0]  # w1 for x1, w2 for x2
        self.bias = 0.0
        self.step_count = 0
        self.loss_history: List[float] = []
        self.is_trained = False

    def _sigmoid(self, z: float) -> float:
        z_clamped = max(-20.0, min(20.0, z))
        return 1.0 / (1.0 + math.exp(-z_clamped))

    def _forward(self, x1: float, x2: float) -> float:
        z = self.weights[0] * x1 + self.weights[1] * x2 + self.bias
        return self._sigmoid(z)

    def train_smoke_run(self, train_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Executes bounded CPU smoke training on at most 32 samples for <= 20 steps.
        """
        if len(train_records) > 32:
            raise ValueError(f"Sample count {len(train_records)} exceeds smoke run bound of 32.")

        self.step_count = 0
        self.loss_history = []

        # Single epoch of gradient updates bounded by max_steps
        for r in train_records:
            if self.step_count >= self.max_steps:
                break

            x1 = float(r["x1"])
            x2 = float(r["x2"])
            y = float(r["label"])

            # Forward
            p = self._forward(x1, x2)

            # Binary cross-entropy loss
            eps = 1e-12
            p_safe = max(eps, min(1.0 - eps, p))
            loss = -(y * math.log(p_safe) + (1.0 - y) * math.log(1.0 - p_safe))
            self.loss_history.append(float(loss))

            # Backward update
            grad_z = p - y
            self.weights[0] -= self.learning_rate * grad_z * x1
            self.weights[1] -= self.learning_rate * grad_z * x2
            self.bias -= self.learning_rate * grad_z

            self.step_count += 1

        self.is_trained = True
        avg_loss = sum(self.loss_history) / len(self.loss_history) if self.loss_history else 0.0

        return {
            "model_type": "Deterministic CPU Linear Smoke Baseline",
            "device": "cpu",
            "samples_processed": len(train_records),
            "step_count": self.step_count,
            "final_step_loss": self.loss_history[-1] if self.loss_history else 0.0,
            "average_loss": float(avg_loss),
            "weights": list(self.weights),
            "bias": float(self.bias),
            "notice": "CPU smoke run for interface correctness verification; does not claim useful real-world accuracy."
        }

    def predict_proba(self, records: List[Dict[str, Any]]) -> List[float]:
        if not self.is_trained:
            raise RuntimeError("Model must be trained before calling predict.")
        return [self._forward(float(r["x1"]), float(r["x2"])) for r in records]

    def predict(self, records: List[Dict[str, Any]], threshold: float = 0.5) -> List[int]:
        probs = self.predict_proba(records)
        return [1 if p >= threshold else 0 for p in probs]


def load_fixture(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
