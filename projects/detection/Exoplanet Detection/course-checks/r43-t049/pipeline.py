"""
Input perturbation and robustness testing pipeline for Exoplanet Detection.
Tests interface invariance under stable ID record-order permutation,
and guarantees rejection or explicit fallback mapping for malformed/missing fields.
"""

from typing import Any, Dict, List, Optional
import json


class PerturbationRejectionError(ValueError):
    """Raised when an unsupported or malformed perturbed input is submitted."""
    pass


class RobustClassificationInterface:
    """
    Classification interface with explicit perturbation handling:
    - Maintains stable ID alignment across arbitrary record permutations.
    - Rejects invalid inputs or maps them to fallback if configured.
    """
    def __init__(self, threshold: float = 3.5, fallback_prediction: Optional[int] = None):
        self.threshold = threshold
        self.fallback_prediction = fallback_prediction

    def predict_record(self, record: Dict[str, Any]) -> int:
        if "record_id" not in record or not record["record_id"]:
            raise PerturbationRejectionError("Record missing required 'record_id'.")

        if "x1" not in record or record["x1"] is None or not isinstance(record["x1"], (int, float)):
            if self.fallback_prediction is not None:
                return self.fallback_prediction
            raise PerturbationRejectionError(f"Missing or malformed field 'x1' in record {record.get('record_id')}")

        if "x2" not in record or record["x2"] is None or not isinstance(record["x2"], (int, float)):
            if self.fallback_prediction is not None:
                return self.fallback_prediction
            raise PerturbationRejectionError(f"Missing or malformed field 'x2' in record {record.get('record_id')}")

        # Deterministic boundary: class 1 if x1 >= threshold else 0
        return 1 if float(record["x1"]) >= self.threshold else 0

    def predict_batch_aligned(self, records: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Returns predictions keyed by record_id, preserving ID alignment
        regardless of input order permutation.
        """
        predictions_by_id = {}
        for r in records:
            rec_id = str(r["record_id"])
            pred = self.predict_record(r)
            predictions_by_id[rec_id] = pred
        return predictions_by_id


def load_fixture(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
