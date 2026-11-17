"""
R43-T008 — Error-Analysis Report: Flood Prediction
NOTE: SYNTHETIC FIXTURES for pipeline verification only. NOT real flood data.
"""

FIXTURE_RECORDS = [
    {"record_id":"f0","group_id":"g0","x1":0,"x2":1,"label":0},
    {"record_id":"f1","group_id":"g1","x1":1,"x2":1,"label":0},
    {"record_id":"f2","group_id":"g2","x1":2,"x2":1,"label":0},
    {"record_id":"f3","group_id":"g3","x1":3,"x2":1,"label":0},
    {"record_id":"f4","group_id":"g4","x1":4,"x2":1,"label":1},
    {"record_id":"f5","group_id":"g5","x1":5,"x2":1,"label":1},
    {"record_id":"f6","group_id":"g6","x1":6,"x2":1,"label":1},
    {"record_id":"f7","group_id":"g7","x1":7,"x2":1,"label":1},
]

TRAIN_IDS = ["f0","f1","f2","f4","f5"]
VAL_IDS   = ["f3"]
TEST_IDS  = ["f6","f7"]

REASON_CATEGORIES = ["false_positive","false_negative","correct"]


def classify_error(truth, pred):
    if truth == pred:
        return "correct"
    elif pred == 1 and truth == 0:
        return "false_positive"
    else:
        return "false_negative"


def build_error_report(records, predictions):
    """Build error report with record_id, expected, predicted, reason_category."""
    if len(records) != len(predictions):
        raise ValueError("records and predictions must have equal length")
    report = []
    for record, pred in zip(records, predictions):
        truth = record["label"]
        report.append({
            "record_id": record["record_id"],
            "expected": truth,
            "predicted": pred,
            "reason_category": classify_error(truth, pred),
        })
    return report


def aggregate_report(report):
    counts = {r: 0 for r in REASON_CATEGORIES}
    for row in report:
        counts[row["reason_category"]] += 1
    return counts


def inject_wrong_prediction(records, target_id, wrong_label):
    """Return predictions list with one deliberate error at target_id."""
    preds = [r["label"] for r in records]
    for i, r in enumerate(records):
        if r["record_id"] == target_id:
            preds[i] = wrong_label
            break
    return preds
