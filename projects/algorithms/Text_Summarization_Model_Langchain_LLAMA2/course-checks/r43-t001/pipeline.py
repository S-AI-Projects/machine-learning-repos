"""
R43-T001 — Schema Validator: Text Summarization Input Contract
NOTE: All records in this module are SYNTHETIC FIXTURES for pipeline
verification only. They are NOT observations from any real dataset and
do NOT constitute evidence of real-world summarization performance.
"""
import re

REQUIRED_FIELDS = {"record_id", "source_text", "expected_first_sentence"}

FIXTURE_RECORDS = [
    {
        "record_id": "r001",
        "source_text": "Ada wrote notes. Bo reviewed them.",
        "expected_first_sentence": "Ada wrote notes.",
    },
    {
        "record_id": "r002",
        "source_text": "One sentence.",
        "expected_first_sentence": "One sentence.",
    },
    {
        "record_id": "r003",
        "source_text": "",
        "expected_first_sentence": "",
    },
]


def validate_records(records):
    """
    Validate a list of summarization input records against the data contract.
    Returns (valid_records, errors) where errors is a list of dicts with
    keys: record_id, field, reason.
    """
    errors = []
    valid = []
    seen_ids = {}

    for i, record in enumerate(records):
        rid = record.get("record_id", f"<missing at index {i}>")

        if rid in seen_ids:
            errors.append({
                "record_id": rid,
                "field": "record_id",
                "reason": f"Duplicate record_id; first seen at index {seen_ids[rid]}",
            })
            continue
        seen_ids[rid] = i

        record_errors = []

        for field in REQUIRED_FIELDS:
            if field not in record:
                record_errors.append({
                    "record_id": rid,
                    "field": field,
                    "reason": "Required field is missing",
                })

        if "record_id" in record:
            if not isinstance(record["record_id"], str) or not record["record_id"].strip():
                record_errors.append({
                    "record_id": rid,
                    "field": "record_id",
                    "reason": f"record_id must be a non-empty string; got {type(record['record_id']).__name__!r}",
                })

        if "source_text" in record:
            if not isinstance(record["source_text"], str):
                record_errors.append({
                    "record_id": rid,
                    "field": "source_text",
                    "reason": f"source_text must be a string; got {type(record['source_text']).__name__!r}",
                })

        if "expected_first_sentence" in record:
            if not isinstance(record["expected_first_sentence"], str):
                record_errors.append({
                    "record_id": rid,
                    "field": "expected_first_sentence",
                    "reason": (
                        f"expected_first_sentence must be a string; "
                        f"got {type(record['expected_first_sentence']).__name__!r}"
                    ),
                })

        if record_errors:
            errors.extend(record_errors)
        else:
            valid.append(record)

    return valid, errors


def extract_first_sentence(text):
    """
    Deterministic first-sentence extractor (no LLM).
    Returns the first sentence terminated by '.', '!' or '?', or the
    full text if no terminator is found. Empty input returns empty string.
    NOTE: This is a labelled baseline rule, NOT an LLM quality claim.
    """
    if not text:
        return ""
    match = re.search(r'[^.!?]*[.!?]', text)
    if match:
        return match.group(0).strip()
    return text.strip()
