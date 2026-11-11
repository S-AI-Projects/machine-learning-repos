"""Small, offline metric and split checks for R43-T014."""


def binary_precision_recall(y_true, y_pred, positive_label=1, zero_division=0.0):
    """Return binary precision/recall for the positive class.

    Averaging is binary: only positive_label is scored. When a denominator is
    zero, return zero_division (0.0 by default).
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")
    if not y_true:
        raise ValueError("at least one label is required")
    if zero_division not in (0.0, 1.0):
        raise ValueError("zero_division must be 0.0 or 1.0")
    if any(value not in (0, 1) for value in y_true + y_pred):
        raise ValueError("labels and predictions must be binary 0 or 1")

    true_positive = sum(
        truth == positive_label and prediction == positive_label
        for truth, prediction in zip(y_true, y_pred)
    )
    false_positive = sum(
        truth != positive_label and prediction == positive_label
        for truth, prediction in zip(y_true, y_pred)
    )
    false_negative = sum(
        truth == positive_label and prediction != positive_label
        for truth, prediction in zip(y_true, y_pred)
    )

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = (
        true_positive / precision_denominator
        if precision_denominator else zero_division
    )
    recall = (
        true_positive / recall_denominator
        if recall_denominator else zero_division
    )
    return {"precision": precision, "recall": recall}


def validate_stratified_splits(records, splits):
    """Check complete, non-overlapping, group-safe splits with both classes."""
    expected_fields = {"record_id", "group_id", "x1", "x2", "label"}
    if any(set(record) != expected_fields for record in records):
        raise ValueError("records must match the documented five-field schema")

    by_id = {record["record_id"]: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("record_id values must be unique")
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("splits must define train, validation, and test")

    assignments = [record_id for ids in splits.values() for record_id in ids]
    if len(assignments) != len(set(assignments)):
        raise ValueError("a record may appear in only one split")
    if set(assignments) != set(by_id):
        raise ValueError("splits must cover every record exactly once")

    groups_by_split = {
        name: {by_id[record_id]["group_id"] for record_id in ids}
        for name, ids in splits.items()
    }
    if (
        groups_by_split["train"] & groups_by_split["validation"]
        or groups_by_split["train"] & groups_by_split["test"]
        or groups_by_split["validation"] & groups_by_split["test"]
    ):
        raise ValueError("groups must not cross splits")

    for ids in splits.values():
        if {by_id[record_id]["label"] for record_id in ids} != {0, 1}:
            raise ValueError("each split must contain both classes")
    return True
