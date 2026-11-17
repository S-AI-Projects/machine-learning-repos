"""
Pipeline schema validator for Grade Prediction classification input fields:
record_id, group_id, x1, x2, label.
Reports record ID, field, and specific failure reason without silently dropping records.
"""

from typing import Any, Dict, List, Optional


class ValidationResult:
    def __init__(self, is_valid: bool, valid_records: List[Dict[str, Any]], errors: List[Dict[str, Any]]):
        self.is_valid = is_valid
        self.valid_records = valid_records
        self.errors = errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "valid_record_count": len(self.valid_records),
            "error_count": len(self.errors),
            "errors": self.errors,
        }


def validate_schema(records: List[Dict[str, Any]]) -> ValidationResult:
    """
    Validates a list of classification input records against the data contract:
    - record_id: non-empty string or integer, must be unique across records
    - group_id: non-empty string or integer
    - x1: numeric (float or int), finite
    - x2: numeric (float or int), finite
    - label: binary integer in {0, 1}
    """
    errors: List[Dict[str, Any]] = []
    valid_records: List[Dict[str, Any]] = []
    seen_record_ids = set()

    for idx, row in enumerate(records):
        row_errors = []
        rec_id = row.get("record_id")

        # 1. Validate record_id presence and format
        if rec_id is None or (isinstance(rec_id, str) and not rec_id.strip()):
            row_errors.append({
                "record_id": f"row_index_{idx}",
                "field": "record_id",
                "reason": "Missing or empty record_id"
            })
            effective_id = f"row_index_{idx}"
        else:
            effective_id = str(rec_id)
            if effective_id in seen_record_ids:
                row_errors.append({
                    "record_id": effective_id,
                    "field": "record_id",
                    "reason": f"Duplicate record_id '{effective_id}' detected"
                })
            else:
                seen_record_ids.add(effective_id)

        # 2. Validate group_id
        if "group_id" not in row or row["group_id"] is None or (isinstance(row["group_id"], str) and not row["group_id"].strip()):
            row_errors.append({
                "record_id": effective_id,
                "field": "group_id",
                "reason": "Missing or empty group_id"
            })

        # 3. Validate x1
        x1_val = row.get("x1")
        if x1_val is None or not isinstance(x1_val, (int, float)) or isinstance(x1_val, bool):
            row_errors.append({
                "record_id": effective_id,
                "field": "x1",
                "reason": f"Malformed x1 value: expected numeric float/int, got {type(x1_val).__name__} ({repr(x1_val)})"
            })

        # 4. Validate x2
        x2_val = row.get("x2")
        if x2_val is None or not isinstance(x2_val, (int, float)) or isinstance(x2_val, bool):
            row_errors.append({
                "record_id": effective_id,
                "field": "x2",
                "reason": f"Malformed x2 value: expected numeric float/int, got {type(x2_val).__name__} ({repr(x2_val)})"
            })

        # 5. Validate label
        lbl = row.get("label")
        if lbl is None or not isinstance(lbl, int) or isinstance(lbl, bool) or lbl not in (0, 1):
            row_errors.append({
                "record_id": effective_id,
                "field": "label",
                "reason": f"Malformed label: expected binary integer in {{0, 1}}, got {repr(lbl)}"
            })

        if row_errors:
            errors.extend(row_errors)
        else:
            valid_records.append(row)

    is_valid = len(errors) == 0
    return ValidationResult(is_valid=is_valid, valid_records=valid_records, errors=errors)
