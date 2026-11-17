# Course Check R43-T025: Explicit Preprocessing

## Overview
This component implements an explicit numeric preprocessor under `projects/prediction/Blood Glucose Level Prediction (Machine Learning)`.
It guarantees:
- Scaling parameters ($\min$, $\max$, scale factors) are fitted exclusively on training rows.
- Data leakage into held-out sets via public `transform()` interface is strictly impossible.
- Feature order (`x1`, `x2`), label sets (`{0, 1}`), and missing values are strictly validated.
- Contaminated inputs (missing fields, unexpected types, invalid labels) are rejected.

## Synthetic Fixture
An explicit 8-row synthetic classification fixture is provided in `fixture.json`:
- `x1 = [0, 1, 2, 3, 4, 5, 6, 7]`
- `x2 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`
- `label = [0, 0, 0, 0, 1, 1, 1, 1]`
- Stable IDs: `rec_0` through `rec_7`, with groups `grp_0` and `grp_1`
- Stratified Split:
  - Train: `["rec_0", "rec_1", "rec_4", "rec_5"]`
  - Test: `["rec_2", "rec_3", "rec_6", "rec_7"]`

*Notice: This fixture is strictly for pipeline preprocessing validation, not real-world blood glucose measurements.*

## Running Tests
Run the offline CPU-only unittest discovery:
```bash
python3 -m unittest discover -s "projects/prediction/Blood Glucose Level Prediction (Machine Learning)/course-checks/r43-t025" -p "test_*.py"
```
