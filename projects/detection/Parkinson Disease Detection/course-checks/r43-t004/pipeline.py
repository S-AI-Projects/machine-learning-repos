"""
R43-T004 — Metric Correctness: Parkinson Disease Detection
NOTE: SYNTHETIC FIXTURES for pipeline verification only. NOT clinical Parkinson data.
"""

FIXTURE_RECORDS = [
    {"record_id": "p0", "group_id": "g0", "x1": 0, "x2": 1, "label": 0},
    {"record_id": "p1", "group_id": "g1", "x1": 1, "x2": 1, "label": 0},
    {"record_id": "p2", "group_id": "g2", "x1": 2, "x2": 1, "label": 0},
    {"record_id": "p3", "group_id": "g3", "x1": 3, "x2": 1, "label": 0},
    {"record_id": "p4", "group_id": "g4", "x1": 4, "x2": 1, "label": 1},
    {"record_id": "p5", "group_id": "g5", "x1": 5, "x2": 1, "label": 1},
    {"record_id": "p6", "group_id": "g6", "x1": 6, "x2": 1, "label": 1},
    {"record_id": "p7", "group_id": "g7", "x1": 7, "x2": 1, "label": 1},
]

TRAIN_IDS = ["p0", "p1", "p2", "p4", "p5"]
VAL_IDS   = ["p3"]
TEST_IDS  = ["p6", "p7"]


def precision(truth, pred, pos_label=1, zero_division=0.0):
    """Precision = TP / (TP + FP). Explicit zero-division handling."""
    tp = sum(1 for t, p in zip(truth, pred) if t == pos_label and p == pos_label)
    fp = sum(1 for t, p in zip(truth, pred) if t != pos_label and p == pos_label)
    denom = tp + fp
    return tp / denom if denom > 0 else zero_division


def recall(truth, pred, pos_label=1, zero_division=0.0):
    """Recall = TP / (TP + FN). Explicit zero-division handling."""
    tp = sum(1 for t, p in zip(truth, pred) if t == pos_label and p == pos_label)
    fn = sum(1 for t, p in zip(truth, pred) if t == pos_label and p != pos_label)
    denom = tp + fn
    return tp / denom if denom > 0 else zero_division


def accuracy(truth, pred):
    """Accuracy = correct / total."""
    if not truth:
        return 0.0
    return sum(t == p for t, p in zip(truth, pred)) / len(truth)


def compute_metrics(truth, pred):
    return {
        "precision": precision(truth, pred),
        "recall": recall(truth, pred),
        "accuracy": accuracy(truth, pred),
    }
