"""Bounded, standard-library classification smoke pipeline."""
from collections import defaultdict
import math
import random

SPLITS = ("train", "validation", "test")
MAX_SAMPLES, MAX_UPDATES = 32, 20

def validate_records(rows):
    if not isinstance(rows, list) or len(rows) != 8:
        raise ValueError("fixture must contain exactly eight rows")
    ids, groups, x_values, labels = set(), set(), set(), {0: 0, 1: 0}
    for row in rows:
        rid, gid = row.get("record_id"), row.get("group_id")
        x1, x2, label = row.get("x1"), row.get("x2"), row.get("label")
        if not rid or rid in ids:
            raise ValueError("record_id values must be present and unique")
        if not gid or gid in groups:
            raise ValueError("group_id values must be present and unique")
        if isinstance(x1, bool) or not isinstance(x1, (int, float)) or not math.isfinite(x1):
            raise ValueError("x1 must be a finite number")
        if x2 != 1:
            raise ValueError("x2 must equal 1")
        if label not in (0, 1):
            raise ValueError("label must be binary")
        ids.add(rid); groups.add(gid); x_values.add(x1); labels[label] += 1
    if x_values != set(range(8)) or labels != {0: 4, 1: 4}:
        raise ValueError("fixture must have x1=0..7 and four rows per class")
    return True

def validate_group_separation(splits):
    if set(splits) != set(SPLITS):
        raise ValueError("train, validation, and test splits are required")
    owners = {}
    for split, rows in splits.items():
        for row in rows:
            group = row.get("group_id")
            previous = owners.setdefault(group, split)
            if previous != split:
                raise ValueError(f"group {group!r} crosses split boundaries")
    return True

def stratified_split(rows, seed=43):
    validate_records(rows)
    classes = defaultdict(list)
    for row in rows:
        classes[row["label"]].append(dict(row))
    rng = random.Random(seed)
    result = {split: [] for split in SPLITS}
    for label in sorted(classes):
        group = sorted(classes[label], key=lambda row: row["record_id"])
        rng.shuffle(group)
        result["train"].extend(group[:2])
        result["validation"].extend(group[2:3])
        result["test"].extend(group[3:4])
    for split in SPLITS:
        result[split].sort(key=lambda row: row["record_id"])
    validate_group_separation(result)
    return result

def _sigmoid(value):
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)

class TinyLogisticBaseline:
    """Small deterministic CPU model for interface smoke testing only."""
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.weights, self.bias = [0.0, 0.0], 0.0
        self.steps, self.epochs, self.training_loss = 0, 0, None

    @staticmethod
    def _features(row):
        return float(row["x1"]) / 7.0, float(row["x2"])

    def _probability(self, row):
        score = self.bias + sum(w*x for w, x in zip(self.weights, self._features(row)))
        return _sigmoid(score)

    def fit(self, training_rows, epochs=1, max_updates=MAX_UPDATES):
        if epochs != 1:
            raise ValueError("smoke run is limited to one training epoch")
        if not 1 <= max_updates <= MAX_UPDATES:
            raise ValueError(f"max_updates must be between 1 and {MAX_UPDATES}")
        if not training_rows or len(training_rows) > MAX_SAMPLES:
            raise ValueError(f"training data must contain 1..{MAX_SAMPLES} rows")
        if len(training_rows) * epochs > max_updates:
            raise ValueError(f"training would exceed {max_updates} update steps")
        losses = []
        for row in training_rows:
            y, p = row["label"], self._probability(row)
            q = min(max(p, 1e-12), 1.0 - 1e-12)
            losses.append(-(y*math.log(q) + (1-y)*math.log(1-q)))
            error, features = p-y, self._features(row)
            self.weights = [w-self.learning_rate*error*x for w, x in zip(self.weights, features)]
            self.bias -= self.learning_rate * error
            self.steps += 1
        self.epochs = epochs
        self.training_loss = sum(losses) / len(losses)
        if not math.isfinite(self.training_loss):
            raise ValueError("training loss must be finite")
        return self

    def predict(self, rows):
        if len(rows) > MAX_SAMPLES:
            raise ValueError(f"inference is limited to {MAX_SAMPLES} rows")
        return [[self._probability(row)] for row in rows]

def run_smoke(rows, seed=43):
    if len(rows) > MAX_SAMPLES:
        raise ValueError(f"smoke run is limited to {MAX_SAMPLES} samples")
    splits = stratified_split(rows, seed)
    model = TinyLogisticBaseline().fit(splits["train"], epochs=1, max_updates=MAX_UPDATES)
    validation = model.predict(splits["validation"])
    test = model.predict(splits["test"])
    return {
        "seed": seed, "epochs": model.epochs, "steps": model.steps,
        "train_loss": model.training_loss,
        "split_ids": {name: [r["record_id"] for r in splits[name]] for name in SPLITS},
        "validation_prediction_shape": [len(validation), len(validation[0])],
        "validation_predictions": validation,
        "test_prediction_shape": [len(test), len(test[0])],
        "test_predictions": test,
    }
