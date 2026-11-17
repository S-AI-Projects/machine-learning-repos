"""
R43-T006 — CPU Smoke Run: Sign Language Detection (Text Pipeline)
NOTE: SYNTHETIC FIXTURES for pipeline verification only. NOT real sign language data.
"""
import re

RANDOM_SEED = 42
MAX_SAMPLES = 32
MAX_EPOCHS  = 1
MAX_STEPS   = 20

FIXTURE_RECORDS = [
    {"record_id": "t1", "text": "red apple",   "label": 1},
    {"record_id": "t2", "text": "green apple", "label": 1},
    {"record_id": "t3", "text": "blue sky",    "label": 0},
    {"record_id": "t4", "text": "dark sky",    "label": 0},
    {"record_id": "t5", "text": "red fruit",   "label": 1},
    {"record_id": "t6", "text": "blue cloud",  "label": 0},
]

TRAIN_IDS = ["t1", "t2", "t3", "t4"]
TEST_IDS  = ["t5", "t6"]


def tokenize(text):
    return re.findall(r'\w+', text.lower())


class BagOfWordsBaseline:
    """Deterministic majority-class text baseline. CPU-only, no GPU, no credentials."""
    def __init__(self, seed=RANDOM_SEED):
        self._seed = seed; self._majority = None; self._step_count = 0; self._fitted = False

    def fit(self, records, max_steps=MAX_STEPS):
        assert len(records) <= MAX_SAMPLES
        labels = [r["label"] for r in records]
        self._majority = max(set(labels), key=labels.count)
        self._step_count = min(len(records), max_steps)
        self._fitted = True; return self

    def predict(self, records):
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
        return [self._majority] * len(records)

    @property
    def step_count(self):
        return self._step_count


def run_smoke(records_train, records_test, max_steps=MAX_STEPS):
    model = BagOfWordsBaseline(seed=RANDOM_SEED).fit(records_train, max_steps=max_steps)
    preds = model.predict(records_test)
    return {
        "predictions": preds,
        "step_count": model.step_count,
        "n_test": len(records_test),
        "majority_class": model._majority,
        "is_finite": all(isinstance(p, int) for p in preds),
    }
