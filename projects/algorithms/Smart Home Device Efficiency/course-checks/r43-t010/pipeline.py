"""
R43-T010 — Saved-Pipeline Interface: Smart Home Device Efficiency
NOTE: SYNTHETIC FIXTURES for pipeline verification only.
Fixture: 6 rows, target = 2*x1 + 1 (deterministic formula).
NOT real smart home data.
"""
import json

SCHEMA_VERSION = "1.0.0"
TOLERANCE = 1e-9

FIXTURE_RECORDS = [
    {"record_id":"r1","x1":0,"x2":1,"target":1},
    {"record_id":"r2","x1":1,"x2":1,"target":3},
    {"record_id":"r3","x1":2,"x2":1,"target":5},
    {"record_id":"r4","x1":3,"x2":1,"target":7},
    {"record_id":"r5","x1":4,"x2":1,"target":9},
    {"record_id":"r6","x1":5,"x2":1,"target":11},
]

TRAIN_IDS = ["r1","r2","r3","r4"]
VAL_IDS   = ["r5"]
TEST_IDS  = ["r6"]


class LinearRegressionFixture:
    """Fits y = slope*x1 + intercept on training data via least squares (no libraries)."""
    def __init__(self):
        self._slope = None; self._intercept = None; self._fitted = False

    def fit(self, records):
        x = [r["x1"] for r in records]; y = [r["target"] for r in records]
        n = len(x)
        x_mean = sum(x)/n; y_mean = sum(y)/n
        num = sum((xi - x_mean)*(yi - y_mean) for xi, yi in zip(x, y))
        den = sum((xi - x_mean)**2 for xi in x)
        self._slope = num/den if den != 0 else 0.0
        self._intercept = y_mean - self._slope * x_mean
        self._fitted = True; return self

    def predict(self, records):
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
        self._validate_schema(records)
        return [self._slope * r["x1"] + self._intercept for r in records]

    def _validate_schema(self, records):
        for r in records:
            if "x1" not in r:
                raise ValueError(f"Missing 'x1' in record {r.get('record_id')!r}")

    def to_json(self):
        if not self._fitted:
            raise RuntimeError("Not fitted yet")
        return json.dumps({
            "schema_version": SCHEMA_VERSION,
            "slope": self._slope,
            "intercept": self._intercept,
        })

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Incompatible schema_version: {data.get('schema_version')!r}")
        if "slope" not in data or "intercept" not in data:
            raise ValueError("Missing 'slope' or 'intercept' in saved model")
        obj = cls()
        obj._slope = data["slope"]; obj._intercept = data["intercept"]; obj._fitted = True
        return obj
