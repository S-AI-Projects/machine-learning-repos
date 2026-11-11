"""Small deterministic error analysis for an explicitly synthetic fixture."""
from __future__ import annotations
FEATURES = ("x1", "x2")
SPLITS = ("train", "validation", "test")

def validate_fixture(fixture):
    if not isinstance(fixture, dict) or fixture.get("schema_version") != 1:
        raise ValueError("unsupported fixture schema")
    if fixture.get("seed") != 23 or fixture.get("feature_order") != list(FEATURES):
        raise ValueError("unexpected fixture configuration")
    rows = fixture.get("records")
    if not isinstance(rows, list) or len(rows) != 8:
        raise ValueError("fixture must contain eight rows")
    ids = set()
    for row in rows:
        if set(row) != {"record_id", "group_id", "x1", "x2", "label"}:
            raise ValueError("record fields do not match the schema")
        if not row["record_id"] or not row["group_id"] or row["record_id"] in ids:
            raise ValueError("record IDs must be unique and IDs nonempty")
        ids.add(row["record_id"])
        if type(row["x1"]) is not int or row["x2"] != 1 or row["label"] not in (0, 1):
            raise ValueError("invalid fixture values")
    splits = fixture.get("split_ids")
    if not isinstance(splits, dict) or set(splits) != set(SPLITS):
        raise ValueError("split names are invalid")
    flat = [rid for split in SPLITS for rid in splits[split]]
    if len(flat) != len(set(flat)) or set(flat) != ids:
        raise ValueError("split IDs must partition fixture records")
    groups = {r["record_id"]: r["group_id"] for r in rows}
    for group in set(groups.values()):
        owners = {split for split in SPLITS for rid in splits[split] if groups[rid] == group}
        if len(owners) != 1:
            raise ValueError("a group crosses splits")
    return rows

def analyze(fixture, prediction_data):
    rows = validate_fixture(fixture)
    if not isinstance(prediction_data, dict) or set(prediction_data) != {"predictions", "reason_categories"}:
        raise ValueError("prediction input fields are invalid")
    predicted = prediction_data["predictions"]
    reasons = prediction_data["reason_categories"]
    ids = {r["record_id"] for r in rows}
    if not isinstance(predicted, dict) or set(predicted) != ids:
        raise ValueError("predictions must cover every record exactly once")
    if not isinstance(reasons, dict) or not set(reasons) <= ids:
        raise ValueError("reason categories contain unknown record IDs")
    for value in predicted.values():
        if type(value) is not int or value not in (0, 1):
            raise ValueError("predictions must be binary integers")
    errors = []
    for row in rows:
        rid = row["record_id"]
        guess = predicted[rid]
        if guess != row["label"]:
            reason = reasons.get(rid)
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("every incorrect prediction needs a reason category")
            errors.append({"record_id": rid, "expected": row["label"], "predicted": guess, "reason_category": reason})
    if set(reasons) != {item["record_id"] for item in errors}:
        raise ValueError("reason categories must correspond exactly to incorrect predictions")
    counts = {"total": len(rows), "correct": len(rows) - len(errors), "errors": len(errors)}
    by_split = {}
    for split in SPLITS:
        split_errors = sum(item["record_id"] in fixture["split_ids"][split] for item in errors)
        by_split[split] = {"total": len(fixture["split_ids"][split]), "errors": split_errors}
    return {"counts": counts, "by_split": by_split, "errors": errors}
