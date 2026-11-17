# Course Check R43-T030: Saved Pipeline Interface

## Overview
This component implements a secure saved-pipeline interface under `projects/prediction/Health Insurance Price prediction`.
It guarantees:
- Secure parameter persistence using explicit JSON format (avoiding unsafe pickle deserialization).
- Prediction consistency: Pre-save and post-reload predictions agree within explicit zero/epsilon tolerance.
- Complete autonomy at inference time: Requires no training data and no remote credentials.
- Strict schema and version validation at prediction time and model load time.

## Synthetic Fixture
An explicit 8-row synthetic classification fixture is provided in `fixture.json`:
- `x1 = [0, 1, 2, 3, 4, 5, 6, 7]`
- `x2 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`
- `label = [0, 0, 0, 0, 1, 1, 1, 1]`
- Stable IDs: `rec_0` through `rec_7`, with groups `grp_0` and `grp_1`
- Split:
  - Train: `["rec_0", "rec_1", "rec_4", "rec_5"]`
  - Test: `["rec_2", "rec_3", "rec_6", "rec_7"]`

## Running Tests
Run the offline CPU-only unittest discovery:
```bash
python3 -m unittest discover -s "projects/prediction/Health Insurance Price prediction/course-checks/r43-t030" -p "test_*.py"
```
