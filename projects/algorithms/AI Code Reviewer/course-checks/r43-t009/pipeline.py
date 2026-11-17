"""
R43-T009 — Robustness Fixture: AI Code Reviewer
NOTE: SYNTHETIC FIXTURES for pipeline verification only. NOT real code review data.
"""

FIXTURE_RECORDS = [
    {"record_id":"a0","group_id":"g0","x1":0,"x2":1,"label":0},
    {"record_id":"a1","group_id":"g1","x1":1,"x2":1,"label":0},
    {"record_id":"a2","group_id":"g2","x1":2,"x2":1,"label":0},
    {"record_id":"a3","group_id":"g3","x1":3,"x2":1,"label":0},
    {"record_id":"a4","group_id":"g4","x1":4,"x2":1,"label":1},
    {"record_id":"a5","group_id":"g5","x1":5,"x2":1,"label":1},
    {"record_id":"a6","group_id":"g6","x1":6,"x2":1,"label":1},
    {"record_id":"a7","group_id":"g7","x1":7,"x2":1,"label":1},
]

TRAIN_IDS = ["a0","a1","a2","a4","a5"]
VAL_IDS   = ["a3"]
TEST_IDS  = ["a6","a7"]


class MajorityClassifier:
    def __init__(self):
        self._majority = None

    def fit(self, records):
        labels = [r["label"] for r in records]
        self._majority = max(set(labels), key=labels.count)
        return self

    def predict(self, records):
        if self._majority is None:
            raise RuntimeError("Call fit() before predict()")
        for r in records:
            if "x1" not in r or "x2" not in r:
                raise ValueError(f"Missing required field in record {r.get('record_id')!r}")
        return [self._majority] * len(records)


def shuffle_records_by_id(records):
    """Return records sorted by record_id (stable, deterministic). Does NOT randomize time order."""
    return sorted(records, key=lambda r: r["record_id"])


def inject_missing_field(records, target_id, field):
    """Return copy of records with one field removed from target record."""
    result = []
    for r in records:
        rc = dict(r)
        if rc["record_id"] == target_id:
            rc.pop(field, None)
        result.append(rc)
    return result
