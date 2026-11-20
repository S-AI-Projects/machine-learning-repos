"""
Reproducible configuration manager and deterministic pipeline for Retail Price Prediction.
Serializes seed, schema version, feature ordering, split IDs, and algorithm settings.
Rejects unknown fields, contradictory settings, and incompatible input shapes.
"""

import json
import platform
import sys
from typing import Any, Dict, List, Optional


class ConfigurationError(ValueError):
    """Raised when configuration contains unknown or contradictory settings."""
    pass


ALLOWED_CONFIG_KEYS = {
    "schema_version",
    "seed",
    "feature_ordering",
    "split_ids",
    "algorithm_settings",
    "environment_metadata"
}

ALLOWED_ALGORITHMS = {"logistic_regression", "naive_bayes", "linear_threshold"}


def create_default_config() -> Dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "seed": 42,
        "feature_ordering": ["x1", "x2"],
        "split_ids": {
            "train": ["r1", "r2", "r5", "r6"],
            "test": ["r3", "r4", "r7", "r8"]
        },
        "algorithm_settings": {
            "algorithm": "linear_threshold",
            "threshold": 3.5
        },
        "environment_metadata": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform()
        }
    }


def validate_config(config: Dict[str, Any]) -> None:
    # 1. Reject unknown top-level keys
    extra_keys = set(config.keys()) - ALLOWED_CONFIG_KEYS
    if extra_keys:
        raise ConfigurationError(f"Unknown configuration field(s): {extra_keys}")

    # 2. Validate schema_version
    version = config.get("schema_version")
    if not isinstance(version, str) or not version.strip():
        raise ConfigurationError("Missing or invalid 'schema_version'")

    # 3. Validate seed
    seed = config.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ConfigurationError(f"Contradictory/invalid seed: expected non-negative int, got {seed}")

    # 4. Validate feature_ordering
    features = config.get("feature_ordering")
    if not isinstance(features, list) or features != ["x1", "x2"]:
        raise ConfigurationError(f"Invalid feature_ordering: expected ['x1', 'x2'], got {features}")

    # 5. Validate split_ids
    splits = config.get("split_ids", {})
    if not isinstance(splits, dict) or "train" not in splits or "test" not in splits:
        raise ConfigurationError("Invalid split_ids: must contain 'train' and 'test'")
    train_set = set(splits["train"])
    test_set = set(splits["test"])
    if train_set.intersection(test_set):
        raise ConfigurationError(f"Contradictory split_ids: train and test overlap on {train_set.intersection(test_set)}")

    # 6. Validate algorithm_settings
    algo_settings = config.get("algorithm_settings", {})
    if not isinstance(algo_settings, dict):
        raise ConfigurationError("Missing or invalid 'algorithm_settings'")
    algo_name = algo_settings.get("algorithm")
    if algo_name not in ALLOWED_ALGORITHMS:
        raise ConfigurationError(f"Unknown or unsupported algorithm: {algo_name}")


class ReproduciblePipeline:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config if config is not None else create_default_config()
        validate_config(self.config)

    def serialize_config(self) -> str:
        return json.dumps(self.config, indent=2)

    @classmethod
    def from_serialized_config(cls, json_str: str) -> "ReproduciblePipeline":
        try:
            data = json.loads(json_str)
        except Exception as e:
            raise ConfigurationError(f"Malformed JSON configuration: {e}")
        return cls(data)

    def predict(self, records: List[Dict[str, Any]]) -> List[int]:
        """
        Executes prediction using the configured feature ordering and algorithm settings.
        Validates input shape and required features.
        """
        expected_features = self.config["feature_ordering"]
        threshold = self.config["algorithm_settings"].get("threshold", 3.5)
        predictions: List[int] = []

        for r in records:
            for feat in expected_features:
                if feat not in r:
                    raise ConfigurationError(f"Incompatible input shape: missing feature '{feat}' in record {r.get('record_id')}")
            
            x1 = float(r["x1"])
            pred = 1 if x1 >= threshold else 0
            predictions.append(pred)

        return predictions


def load_fixture(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
