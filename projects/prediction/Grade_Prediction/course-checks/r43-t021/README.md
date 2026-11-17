# Course Check R43-T021: Grade Prediction Schema Validator

## Overview
This component implements a strict schema validator for classification input records under `projects/prediction/Grade_Prediction`.
It validates the input contract:
- `record_id`: Non-empty string or integer, strictly unique across the dataset
- `group_id`: Non-empty string or integer representing group isolation
- `x1`: Numeric value (`float` or `int`), finite
- `x2`: Numeric value (`float` or `int`), finite
- `label`: Binary classification target in `{0, 1}`

Failures are reported with explicit details (`record_id`, `field`, and `reason`) without silently dropping records.

## Synthetic Fixture
An explicit 8-row synthetic fixture is provided in `fixture.json`:
- `x1 = [0, 1, 2, 3, 4, 5, 6, 7]`
- `x2 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`
- `label = [0, 0, 0, 0, 1, 1, 1, 1]`
- Stable IDs: `rec_0` through `rec_7`, with groups `grp_0` and `grp_1`
- Fixed stratified split:
  - Train: `["rec_0", "rec_1", "rec_4", "rec_5"]`
  - Test: `["rec_2", "rec_3", "rec_6", "rec_7"]`

*Notice: This fixture is strictly for pipeline schema and verification testing, not real-world dataset observations.*

## Running Tests
Run the offline CPU-only unittest discovery:
```bash
python3 -m unittest discover -s projects/prediction/Grade_Prediction/course-checks/r43-t021 -p "test_*.py"
```
