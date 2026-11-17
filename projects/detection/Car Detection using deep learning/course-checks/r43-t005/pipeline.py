"""
R43-T005 — Preprocessing Contract: Car Detection using Deep Learning
NOTE: SYNTHETIC FIXTURES for pipeline verification only.
"""

SCHEMA_FEATURES = ["x1", "x2"]
SCHEMA_LABELS   = [0, 1]

FIXTURE_RECORDS = [
    {"record_id": "c0", "group_id": "g0", "x1": 0, "x2": 1, "label": 0},
    {"record_id": "c1", "group_id": "g1", "x1": 1, "x2": 1, "label": 0},
    {"record_id": "c2", "group_id": "g2", "x1": 2, "x2": 1, "label": 0},
    {"record_id": "c3", "group_id": "g3", "x1": 3, "x2": 1, "label": 0},
    {"record_id": "c4", "group_id": "g4", "x1": 4, "x2": 1, "label": 1},
    {"record_id": "c5", "group_id": "g5", "x1": 5, "x2": 1, "label": 1},
    {"record_id": "c6", "group_id": "g6", "x1": 6, "x2": 1, "label": 1},
    {"record_id": "c7", "group_id": "g7", "x1": 7, "x2": 1, "label": 1},
]

TRAIN_IDS = ["c0", "c1", "c2", "c4", "c5"]
VAL_IDS   = ["c3"]
TEST_IDS  = ["c6", "c7"]


class MinMaxScaler:
    """Fits numeric scaler on training rows only. Validates feature order and label set."""
    def __init__(self, feature_names=None):
        self._feature_names = feature_names or SCHEMA_FEATURES
        self._min = {}; self._max = {}; self._fitted = False

    def fit(self, records):
        for feat in self._feature_names:
            vals = [r[feat] for r in records if r.get(feat) is not None]
            if not vals:
                raise ValueError(f"All values missing for feature '{feat}'")
            self._min[feat] = min(vals); self._max[feat] = max(vals)
        self._fitted = True; return self

    def transform(self, records):
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")
        result = []
        for r in records:
            self._validate_record(r)
            row = {}
            for feat in self._feature_names:
                val = r.get(feat)
                if val is None:
                    raise ValueError(
                        f"Missing value for feature '{feat}' in record '{r.get('record_id')}'. "
                        "Policy: raise on missing."
                    )
                rng = self._max[feat] - self._min[feat]
                row[feat] = (val - self._min[feat]) / rng if rng != 0 else 0.0
            result.append(row)
        return result

    def _validate_record(self, record):
        for feat in self._feature_names:
            if feat not in record:
                raise ValueError(f"Missing required feature '{feat}' in record {record.get('record_id')!r}")
        if "label" in record and record["label"] not in SCHEMA_LABELS:
            raise ValueError(f"Unknown label {record['label']!r}; expected {SCHEMA_LABELS}")
