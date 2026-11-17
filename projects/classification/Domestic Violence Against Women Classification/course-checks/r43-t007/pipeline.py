"""
R43-T007 — Configuration Reproducibility: Domestic Violence Against Women Classification
NOTE: SYNTHETIC FIXTURES for pipeline verification only. NOT real domestic violence data.
"""
import json, sys

CONFIG_SCHEMA_VERSION = "1.0.0"
KNOWN_ALGORITHMS = ["majority_class", "threshold"]
REQUIRED_CONFIG_KEYS = {"schema_version","seed","feature_order","split_ids","algorithm","algorithm_settings"}

FIXTURE_RECORDS = [
    {"record_id": "d0","group_id":"g0","x1":0,"x2":1,"label":0},
    {"record_id": "d1","group_id":"g1","x1":1,"x2":1,"label":0},
    {"record_id": "d2","group_id":"g2","x1":2,"x2":1,"label":0},
    {"record_id": "d3","group_id":"g3","x1":3,"x2":1,"label":0},
    {"record_id": "d4","group_id":"g4","x1":4,"x2":1,"label":1},
    {"record_id": "d5","group_id":"g5","x1":5,"x2":1,"label":1},
    {"record_id": "d6","group_id":"g6","x1":6,"x2":1,"label":1},
    {"record_id": "d7","group_id":"g7","x1":7,"x2":1,"label":1},
]

DEFAULT_CONFIG = {
    "schema_version": CONFIG_SCHEMA_VERSION,
    "seed": 42,
    "feature_order": ["x1","x2"],
    "split_ids": {
        "train": ["d0","d1","d2","d4","d5"],
        "validation": ["d3"],
        "test": ["d6","d7"],
    },
    "algorithm": "majority_class",
    "algorithm_settings": {},
    "environment": {"python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"},
}


def validate_config(config):
    missing = REQUIRED_CONFIG_KEYS - set(config.keys())
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")
    unknown = set(config.keys()) - (REQUIRED_CONFIG_KEYS | {"environment"})
    if unknown:
        raise ValueError(f"Config has unknown keys: {unknown}")
    if config["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"Incompatible schema_version {config['schema_version']!r}")
    if config["algorithm"] not in KNOWN_ALGORITHMS:
        raise ValueError(f"Unknown algorithm {config['algorithm']!r}")
    if not isinstance(config["feature_order"], list) or not config["feature_order"]:
        raise ValueError("feature_order must be a non-empty list")
    return True


def config_to_json(config):
    return json.dumps(config, sort_keys=True)


def config_from_json(json_str):
    config = json.loads(json_str)
    validate_config(config)
    return config


class MajorityClassPipeline:
    def __init__(self, config):
        validate_config(config); self._config = config; self._majority = None

    def fit(self, records):
        train_ids = set(self._config["split_ids"]["train"])
        train = [r for r in records if r["record_id"] in train_ids]
        labels = [r["label"] for r in train]
        self._majority = max(set(labels), key=labels.count); return self

    def predict(self, records):
        if self._majority is None:
            raise RuntimeError("Call fit() first")
        return [self._majority] * len(records)
