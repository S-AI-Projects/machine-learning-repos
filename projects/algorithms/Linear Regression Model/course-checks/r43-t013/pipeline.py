"""Training-mean regression baseline for the R43-T013 synthetic fixture."""

from math import isfinite
from numbers import Real


def fit_training_mean(records):
    """Fit a constant predictor using targets from training records only."""
    targets = []
    for record in records:
        if record.get("split") != "train":
            continue
        if "target" not in record:
            raise ValueError("training records must include a target")
        target = record["target"]
        if isinstance(target, bool) or not isinstance(target, Real):
            raise TypeError("training targets must be numeric")
        if not isfinite(target):
            raise ValueError("training targets must be finite")
        targets.append(float(target))

    if not targets:
        raise ValueError("at least one training record is required")
    return sum(targets) / len(targets)


def predict_held_out(records, baseline):
    """Return predictions for validation and test records."""
    if isinstance(baseline, bool) or not isinstance(baseline, Real):
        raise TypeError("baseline must be numeric")
    if not isfinite(baseline):
        raise ValueError("baseline must be finite")

    predictions = []
    for record in records:
        if record.get("split") not in {"validation", "test"}:
            continue
        if "target" not in record:
            raise ValueError("held-out records must include a target")
        predictions.append({
            "id": record["id"],
            "split": record["split"],
            "actual": record["target"],
            "prediction": float(baseline),
        })
    return predictions
