"""A tiny offline classification interface used to check row-order robustness."""
from __future__ import annotations
FEATURES = ("x1", "x2")
RECORD_FIELDS = {"record_id", "group_id", "x1", "x2", "label"}

def validate_fixture(fixture):
    if not isinstance(fixture, dict) or fixture.get("schema_version") != 1:
        raise ValueError("unsupported fixture schema")
    if fixture.get("seed") != 19 or fixture.get("feature_order") != list(FEATURES):
        raise ValueError("unexpected fixture configuration")
    rows = fixture.get("records")
    if not isinstance(rows, list) or len(rows) != 8:
        raise ValueError("fixture must contain eight records")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != RECORD_FIELDS:
            raise ValueError("record fields or feature shape do not match schema")
        rid, gid = row["record_id"], row["group_id"]
        if not isinstance(rid, str) or not rid or not isinstance(gid, str) or not gid or rid in seen:
            raise ValueError("record IDs must be unique and IDs nonempty")
        seen.add(rid)
        if type(row["x1"]) is not int or type(row["x2"]) is not int or row["label"] not in (0, 1):
            raise ValueError("record values are invalid")
    splits = fixture.get("split_ids")
    if not isinstance(splits, dict) or set(splits) != {"train", "validation", "test"}:
        raise ValueError("split names are invalid")
    flat = [rid for split in ("train", "validation", "test") for rid in splits[split]]
    if len(flat) != len(set(flat)) or set(flat) != seen:
        raise ValueError("split IDs must partition records")
    groups = {row["record_id"]: row["group_id"] for row in rows}
    for group in set(groups.values()):
        owners = {split for split in splits for rid in splits[split] if groups[rid] == group}
        if len(owners) != 1:
            raise ValueError("a group crosses splits")
    return rows

def predict_by_id(records):
    """Return results keyed by stable ID so input row order cannot affect output."""
    results = {}
    for row in records:
        if not isinstance(row, dict) or set(row) != RECORD_FIELDS:
            raise ValueError("record fields or feature shape do not match schema")
        rid = row["record_id"]
        if not isinstance(rid, str) or not rid or rid in results:
            raise ValueError("record IDs must be unique and nonempty")
        if type(row["x1"]) is not int or type(row["x2"]) is not int or row["label"] not in (0, 1):
            raise ValueError("record values are invalid")
        results[rid] = int(row["x1"] >= 4)
    return results
