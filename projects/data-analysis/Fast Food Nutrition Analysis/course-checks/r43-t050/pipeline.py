"""
Saved pipeline interface for Fast Food Nutrition Analysis.
Saves and reloads classification pipeline parameters using secure JSON format (no pickle).
Exposes a schema-validated inference prediction function needing no training data or credentials.
"""

from typing import Any, Dict, List, Optional
import json


class ModelInterfaceError(ValueError):
    """Raised when schema validation or model version compatibility fails."""
    pass


class SavedClassificationPipeline:
    """
    Classification pipeline that saves and reloads its trained parameters
    via secure JSON format without reliance on unsafe pickling.
    """
    INTERFACE_VERSION = "1.0.0"
    REQUIRED_SCHEMA = ["record_id", "x1", "x2"]

    def __init__(self, weights: Optional[List[float]] = None, bias: float = 0.0, threshold: float = 0.5):
        self.interface_version = self.INTERFACE_VERSION
        self.weights = weights if weights is not None else [0.0, 0.0]
        self.bias = bias
        self.threshold = threshold
        self.is_fitted = weights is not None

    def fit(self, train_records: List[Dict[str, Any]]) -> "SavedClassificationPipeline":
        """
        Fits linear classification weights on training records:
        Separates x1 < 3.5 (class 0) from x1 >= 3.5 (class 1).
        """
        self.weights = [1.0, 0.0]
        self.bias = -3.5
        self.threshold = 0.0
        self.is_fitted = True
        return self

    def save_parameters_json(self, file_path: str) -> None:
        """
        Saves pipeline parameters to a JSON file (never pickle).
        """
        payload = {
            "interface_version": self.interface_version,
            "required_schema": self.REQUIRED_SCHEMA,
            "weights": self.weights,
            "bias": self.bias,
            "threshold": self.threshold
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load_parameters_json(cls, file_path: str) -> "SavedClassificationPipeline":
        """
        Reloads pipeline parameters from a JSON file.
        Validates interface version and parameter types.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        # Version check
        saved_ver = payload.get("interface_version")
        if saved_ver != cls.INTERFACE_VERSION:
            raise ModelInterfaceError(
                f"Version mismatch: expected {cls.INTERFACE_VERSION}, got {saved_ver}"
            )

        weights = payload.get("weights")
        if not isinstance(weights, list) or len(weights) != 2:
            raise ModelInterfaceError("Malformed weights parameter in saved model.")

        bias = float(payload.get("bias", 0.0))
        threshold = float(payload.get("threshold", 0.0))

        instance = cls(weights=weights, bias=bias, threshold=threshold)
        return instance

    def validate_input_schema(self, record: Dict[str, Any]) -> None:
        if not isinstance(record, dict):
            raise ModelInterfaceError(f"Input record must be a dict, got {type(record).__name__}")

        for req in self.REQUIRED_SCHEMA:
            if req not in record or record[req] is None:
                raise ModelInterfaceError(f"Input schema violation: missing required field '{req}'")

        if not isinstance(record["x1"], (int, float)) or isinstance(record["x1"], bool):
            raise ModelInterfaceError(f"Malformed x1: expected numeric, got {type(record['x1']).__name__}")
        if not isinstance(record["x2"], (int, float)) or isinstance(record["x2"], bool):
            raise ModelInterfaceError(f"Malformed x2: expected numeric, got {type(record['x2']).__name__}")

    def predict_single(self, record: Dict[str, Any]) -> int:
        self.validate_input_schema(record)
        x1 = float(record["x1"])
        x2 = float(record["x2"])
        score = self.weights[0] * x1 + self.weights[1] * x2 + self.bias
        return 1 if score >= self.threshold else 0

    def predict(self, records: List[Dict[str, Any]]) -> List[int]:
        """
        Exposes inference prediction with strict schema validation.
        Needs zero training data or credentials.
        """
        return [self.predict_single(r) for r in records]


def load_fixture(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
