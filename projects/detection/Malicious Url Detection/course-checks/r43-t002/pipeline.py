"""
R43-T002 — Leakage-Free Split Manifest: Malicious URL Detection
NOTE: All records are SYNTHETIC FIXTURES for pipeline verification only.
NOT observations from any real malicious-URL dataset.
"""

SCHEMA_FIELDS = ("record_id", "group_id", "x1", "x2", "label")

FIXTURE_RECORDS = [
    {"record_id": "s0", "group_id": "g0", "x1": 0, "x2": 1, "label": 0},
    {"record_id": "s1", "group_id": "g1", "x1": 1, "x2": 1, "label": 0},
    {"record_id": "s2", "group_id": "g2", "x1": 2, "x2": 1, "label": 0},
    {"record_id": "s3", "group_id": "g3", "x1": 3, "x2": 1, "label": 0},
    {"record_id": "s4", "group_id": "g4", "x1": 4, "x2": 1, "label": 1},
    {"record_id": "s5", "group_id": "g5", "x1": 5, "x2": 1, "label": 1},
    {"record_id": "s6", "group_id": "g6", "x1": 6, "x2": 1, "label": 1},
    {"record_id": "s7", "group_id": "g7", "x1": 7, "x2": 1, "label": 1},
]

# Fixed deterministic split (seed=42, stratified by label)
# 60% train, 20% val, 20% test
_SPLIT_MANIFEST = {
    "train": ["s0", "s1", "s2", "s4", "s5"],
    "validation": ["s3"],
    "test": ["s6", "s7"],
}


def get_split_manifest():
    """Return the fixed deterministic split manifest (immutable copy)."""
    return {k: list(v) for k, v in _SPLIT_MANIFEST.items()}


def validate_manifest(manifest):
    """
    Validate that the split manifest has no overlapping IDs and that
    the union covers all records exactly once. Raises ValueError on failure.
    """
    all_ids = set()
    for split_name, ids in manifest.items():
        id_set = set(ids)
        overlap = all_ids & id_set
        if overlap:
            raise ValueError(
                f"Overlap detected between splits: {overlap} appear in multiple splits"
            )
        all_ids |= id_set

    fixture_ids = {r["record_id"] for r in FIXTURE_RECORDS}
    missing = fixture_ids - all_ids
    extra = all_ids - fixture_ids
    if missing:
        raise ValueError(f"Records missing from manifest: {missing}")
    if extra:
        raise ValueError(f"Unknown record IDs in manifest: {extra}")
    return True


def build_split(records, manifest):
    """Return a dict of {split_name: [record, ...]} using the manifest."""
    id_to_record = {r["record_id"]: r for r in records}
    return {
        split_name: [id_to_record[rid] for rid in ids]
        for split_name, ids in manifest.items()
    }
