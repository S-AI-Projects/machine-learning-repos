"""Data-contract validation for the R43-T011 synthetic fixture."""

from collections.abc import Mapping
from typing import Any


REQUIRED_FIELDS = {"record_id", "text", "label"}
ALLOWED_LABELS = {0, 1}


def validate_records(records: Any) -> list[dict[str, str]]:
    """Return all validation errors without dropping or modifying records."""
    errors: list[dict[str, str]] = []
    seen_ids: set[str] = set()

    if not isinstance(records, list):
        return [{
            "record_id": "<input>",
            "field": "records",
            "reason": "Expected a list of records."
        }]

    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            errors.append({
                "record_id": f"<index:{index}>",
                "field": "record",
                "reason": "Expected a mapping/object, not malformed tensor or other data."
            })
            continue

        raw_id = record.get("record_id")
        record_id = (
            raw_id if isinstance(raw_id, str) and raw_id.strip()
            else f"<index:{index}>"
        )

        if "record_id" not in record:
            errors.append({
                "record_id": record_id,
                "field": "record_id",
                "reason": "Required field is missing."
            })
        elif not isinstance(raw_id, str) or not raw_id.strip():
            errors.append({
                "record_id": record_id,
                "field": "record_id",
                "reason": "Expected a non-empty string."
            })
        elif raw_id in seen_ids:
            errors.append({
                "record_id": raw_id,
                "field": "record_id",
                "reason": "Duplicate record ID."
            })
        else:
            seen_ids.add(raw_id)

        for field in ("text", "label"):
            if field not in record:
                errors.append({
                    "record_id": record_id,
                    "field": field,
                    "reason": "Required field is missing."
                })

        text_value = record.get("text")
        if "text" in record and (
            not isinstance(text_value, str) or not text_value.strip()
        ):
            errors.append({
                "record_id": record_id,
                "field": "text",
                "reason": "Expected a non-empty string; numeric, list, mapping and tensor-like values are invalid."
            })

        label_value = record.get("label")
        if "label" in record and (
            isinstance(label_value, bool)
            or not isinstance(label_value, int)
            or label_value not in ALLOWED_LABELS
        ):
            errors.append({
                "record_id": record_id,
                "field": "label",
                "reason": "Expected an integer label of 0 or 1; booleans, floats, strings and tensor-like values are invalid."
            })

        for extra_field in sorted(set(record) - REQUIRED_FIELDS):
            errors.append({
                "record_id": record_id,
                "field": str(extra_field),
                "reason": "Unexpected field is not allowed by the schema."
            })

    return errors



def split_records(records: list[dict], seed: int = 42) -> dict[str, list[dict]]:
    """Create deterministic, non-overlapping train/validation/test partitions."""
    import random

    errors = validate_records(records)
    if errors:
        raise ValueError(f"Cannot split invalid records: {errors}")

    if len(records) < 3:
        raise ValueError("At least three records are required for three splits.")

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)

    train_end = min(max(1, int(len(shuffled) * 0.67)), len(shuffled) - 2)
    validation_end = train_end + 1

    return {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:validation_end],
        "test": shuffled[validation_end:]
    }
