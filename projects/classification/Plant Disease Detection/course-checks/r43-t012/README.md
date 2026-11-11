# R43-T012 — Deterministic Split Manifest

## Purpose

This contribution verifies a deterministic, leakage-aware data split for
the Plant Disease Detection project. It does not reproduce or train the
image-classification model.

## Synthetic fixture

`fixture.json` contains eight synthetic records with stable record IDs,
group IDs, two numeric features, and binary labels.

These records are created only to verify the split pipeline. They are
not observations from the real Plant Disease Detection dataset and
cannot support claims about disease recognition or model performance.

## Split design

- Fixed random seed: 43.
- Split proportions: 50% train, 25% validation, 25% test.
- Labels are stratified at group level.
- Rows sharing `group_id` or `user_id` remain together.
- If a time field is supplied, chronological splitting takes precedence.
- Exact record IDs are written into each split.
- Duplicate record IDs and missing required fields are rejected.

For the eight-record fixture, the expected split sizes are four
training, two validation, and two test records.

## Run the tests

From the repository root:

```bash
python3 -m unittest discover -s 'projects/classification/Plant Disease Detection/course-checks/r43-t012' -p "test_*.py"

python3 'projects/classification/Plant Disease Detection/course-checks/r43-t012/pipeline.py'
