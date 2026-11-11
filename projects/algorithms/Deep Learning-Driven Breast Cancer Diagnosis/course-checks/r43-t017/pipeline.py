"""Validate and execute a tiny deterministic synthetic classification contract."""
from __future__ import annotations
import json
from pathlib import Path

FEATURES = ("x1", "x2")
SPLITS = ("train", "validation", "test")
CONFIG_KEYS = {"schema_version", "seed", "feature_order", "split_ids", "algorithm"}

def load_json(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)

def validate_fixture(fixture):
    if not isinstance(fixture, dict) or fixture.get("schema_version") != 1:
        raise ValueError("unsupported fixture schema")
    if fixture.get("feature_order") != list(FEATURES):
        raise ValueError("fixture feature ordering is incompatible")
    rows = fixture.get("records")
    if not isinstance(rows, list) or len(rows) != 8:
        raise ValueError("fixture must contain eight records")
    ids = set()
    for row in rows:
        if set(row) != {"record_id", "group_id", "x1", "x2", "label"}:
            raise ValueError("record fields do not match the schema")
        if row["record_id"] in ids or not row["record_id"] or not row["group_id"]:
            raise ValueError("record and group IDs must be nonempty and record IDs unique")
        ids.add(row["record_id"])
        if type(row["x1"]) is not int or row["x2"] != 1 or row["label"] not in (0, 1):
            raise ValueError("invalid synthetic record values")
    return rows

def validate_config(config, rows):
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
        raise ValueError("configuration has missing or unknown fields")
    if config["schema_version"] != 1 or type(config["seed"]) is not int:
        raise ValueError("unsupported schema version or seed")
    if config["feature_order"] != list(FEATURES):
        raise ValueError("configuration feature ordering is incompatible")
    algo = config["algorithm"]
    if not isinstance(algo, dict) or set(algo) != {"name", "threshold", "positive_label", "negative_label"}:
        raise ValueError("algorithm settings have missing or unknown fields")
    if algo["name"] != "threshold" or type(algo["threshold"]) not in (int, float):
        raise ValueError("unsupported algorithm settings")
    if (algo["positive_label"], algo["negative_label"]) != (1, 0):
        raise ValueError("contradictory label settings")
    splits = config["split_ids"]
    if not isinstance(splits, dict) or set(splits) != set(SPLITS):
        raise ValueError("split names must be train, validation, and test")
    flattened = [rid for split in SPLITS for rid in splits[split]]
    expected = {r["record_id"] for r in rows}
    if len(flattened) != len(set(flattened)) or set(flattened) != expected:
        raise ValueError("split IDs must partition the fixture exactly once")
    groups = {r["record_id"]: r["group_id"] for r in rows}
    for split in SPLITS:
        if not splits[split]:
            raise ValueError("splits cannot be empty")
    for group in set(groups.values()):
        group_splits = {split for split in SPLITS for rid in splits[split] if groups[rid] == group}
        if len(group_splits) != 1:
            raise ValueError("a group crosses splits")
    return config

def predictions(config, rows):
    validate_config(config, rows)
    algo = config["algorithm"]
    return {r["record_id"]: int(r["x1"] >= algo["threshold"]) for r in rows}

def run(fixture, config):
    rows = validate_fixture(fixture)
    cfg = validate_config(config, rows)
    return {"split_ids": cfg["split_ids"], "predictions": predictions(cfg, rows)}
