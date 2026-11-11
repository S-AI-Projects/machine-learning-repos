# R43-T011: Fake News Detection Data Contract

## Purpose

This folder contains a small, offline, CPU-only data-contract validation
component for the Fake News Detection project.

It validates a synthetic pipeline fixture. It does not reproduce or retrain
the original fake-news detection model.

## Synthetic fixture

`fixture.json` contains six fictional records:

| record_id | text        | label |
|-----------|-------------|-------|
| t1        | red apple   | 1     |
| t2        | green apple | 1     |
| t3        | blue sky    | 0     |
| t4        | dark sky    | 0     |
| t5        | red fruit   | 1     |
| t6        | blue cloud  | 0     |

These records are invented for pipeline verification. They are not a real
news corpus and cannot establish real-world model performance.

## Data contract

The schema is defined in `schema.json`.

- `record_id`: required, non-empty, unique string.
- `text`: required, non-empty string.
- `label`: required integer, either 0 or 1.
- Additional fields are rejected.

Boolean values are not accepted as integer labels. Invalid records produce
errors containing the record ID, field and reason. Records are not silently
discarded.

## Validation and data splitting

`pipeline.py` provides:

- `validate_records(records)` for schema validation.
- `split_records(records, seed=42)` for deterministic, non-overlapping
  train, validation and test partitions.

The six-record fixture produces a small 4/1/1 partition. This is only a
pipeline demonstration, not a statistically meaningful model split.

## Run tests

From the repository root, run:

```bash
python3 -m unittest discover -s projects/nlp/fake-news-detection/course-checks/r43-t011 -p "test_*.py"


```
