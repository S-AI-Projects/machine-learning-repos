"""
R43-T003 — Minimal Baseline: Butterfly Image Classification
NOTE: All records are SYNTHETIC FIXTURES for pipeline verification only.
NOT observations from any real butterfly image dataset.
"""

FIXTURE_RECORDS = [
    {"record_id": "img001", "group_id": "grp0", "pixels": [[0]*4]*4, "label": 0},
    {"record_id": "img002", "group_id": "grp1", "pixels": [[0]*4]*4, "label": 0},
    {"record_id": "img003", "group_id": "grp2", "pixels": [[0]*4]*4, "label": 0},
    {"record_id": "img004", "group_id": "grp3",
     "pixels": [[255,0,0,0],[0,255,0,0],[0,0,255,0],[0,0,0,255]], "label": 1},
    {"record_id": "img005", "group_id": "grp4",
     "pixels": [[255,0,0,0],[0,255,0,0],[0,0,255,0],[0,0,0,255]], "label": 1},
    {"record_id": "img006", "group_id": "grp5",
     "pixels": [[255,0,0,0],[0,255,0,0],[0,0,255,0],[0,0,0,255]], "label": 1},
]
TRAIN_IDS = ["img001", "img002", "img004", "img005"]
VAL_IDS   = ["img003"]
TEST_IDS  = ["img006"]


class MajorityClassBaseline:
    """Majority-class classifier (simplest honest baseline). Fits on training labels only."""
    def __init__(self):
        self._majority_class = None

    def fit(self, records):
        labels = [r["label"] for r in records]
        self._majority_class = max(set(labels), key=labels.count)
        return self

    def predict(self, records):
        if self._majority_class is None:
            raise RuntimeError("Call fit() before predict()")
        return [self._majority_class] * len(records)

    def score(self, records):
        preds = self.predict(records)
        truths = [r["label"] for r in records]
        correct = sum(p == t for p, t in zip(preds, truths))
        return correct / len(truths) if truths else 0.0


def get_split(records, ids):
    id_to_rec = {r["record_id"]: r for r in records}
    return [id_to_rec[i] for i in ids]
