"""JSON-serializable saved prediction interface for a synthetic fixture."""
from __future__ import annotations
import json
import math
from pathlib import Path

FEATURES = ("x1", "x2")
RECORD_FIELDS = {"record_id", "group_id", "x1", "x2", "label"}
PARAMETER_FIELDS = {"schema_version", "pipeline_name", "feature_order", "weights", "bias", "decision_threshold"}

def validate_fixture(fixture):
    if not isinstance(fixture, dict) or fixture.get("schema_version") != 1:
        raise ValueError("unsupported fixture schema version")
    if fixture.get("seed") != 20 or fixture.get("feature_order") != list(FEATURES):
        raise ValueError("fixture configuration mismatch")
    if not str(fixture.get("data_status", "")).startswith("synthetic"):
        raise ValueError("fixture must be explicitly labelled synthetic")
    rows = fixture.get("records")
    if not isinstance(rows, list) or len(rows) != 8:
        raise ValueError("fixture must contain eight records")
    ids, groups = set(), {}
    for row in rows:
        _validate_record(row)
        if row["record_id"] in ids:
            raise ValueError("record IDs must be unique")
        ids.add(row["record_id"])
        groups[row["record_id"]] = row["group_id"]
    splits = fixture.get("split_ids")
    if not isinstance(splits, dict) or set(splits) != {"train", "validation", "test"}:
        raise ValueError("split names are invalid")
    flattened = [rid for name in ("train", "validation", "test") for rid in splits[name]]
    if len(flattened) != len(set(flattened)) or set(flattened) != ids:
        raise ValueError("split IDs must partition the fixture")
    for name, wanted in (("train", 4), ("validation", 2), ("test", 2)):
        labels = [next(r["label"] for r in rows if r["record_id"] == rid) for rid in splits[name]]
        if len(labels) != wanted or set(labels) != {0, 1}:
            raise ValueError("each fixed split must be stratified and have the documented size")
    for group in set(groups.values()):
        owners = {name for name, members in splits.items() if any(groups[rid] == group for rid in members)}
        if len(owners) != 1:
            raise ValueError("a group crosses splits")
    return rows

def _validate_record(row):
    if not isinstance(row, dict) or set(row) != RECORD_FIELDS:
        raise ValueError("record schema mismatch: expected record_id, group_id, x1, x2, label")
    for key in ("record_id", "group_id"):
        if not isinstance(row[key], str) or not row[key]:
            raise ValueError("record and group IDs must be nonempty strings")
    if type(row["x1"]) not in (int, float) or type(row["x2"]) not in (int, float):
        raise ValueError("features must be numeric")
    if not math.isfinite(row["x1"]) or not math.isfinite(row["x2"]):
        raise ValueError("features must be finite")
    if type(row["label"]) is not int or row["label"] not in (0, 1):
        raise ValueError("label must be binary 0 or 1")

def save_parameters(parameters, path):
    _validate_parameters(parameters)
    Path(path).write_text(json.dumps(parameters, indent=2) + "\n", encoding="utf-8")

def load_parameters(path):
    # JSON is used deliberately; never deserialize an untrusted pickle.
    parameters = json.loads(Path(path).read_text(encoding="utf-8"))
    _validate_parameters(parameters)
    return parameters

def _validate_parameters(parameters):
    if not isinstance(parameters, dict) or set(parameters) != PARAMETER_FIELDS:
        raise ValueError("saved pipeline parameter schema mismatch")
    if parameters["schema_version"] != 1 or parameters["feature_order"] != list(FEATURES):
        raise ValueError("unsupported saved pipeline schema/version")
    if not isinstance(parameters["weights"], list) or len(parameters["weights"]) != 2:
        raise ValueError("saved pipeline weights must match the feature schema")
    numbers = parameters["weights"] + [parameters["bias"], parameters["decision_threshold"]]
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in numbers):
        raise ValueError("saved pipeline parameters must be finite numbers")
    if not 0 < parameters["decision_threshold"] < 1:
        raise ValueError("decision threshold must be between zero and one")

def predict(records, parameters):
    _validate_parameters(parameters)
    results = {}
    for row in records:
        _validate_record(row)
        rid = row["record_id"]
        if rid in results:
            raise ValueError("record IDs must be unique")
        z = sum(float(row[name]) * weight for name, weight in zip(FEATURES, parameters["weights"])) + parameters["bias"]
        probability = 1 / (1 + math.exp(-z))
        results[rid] = {"probability": probability, "label": int(probability >= parameters["decision_threshold"])}
    return results
