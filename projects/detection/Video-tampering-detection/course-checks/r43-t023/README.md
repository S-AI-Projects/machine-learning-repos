# Course Check R43-T023: Majority Class Baseline

## Overview
This component implements a majority-class classification baseline under `projects/detection/Video-tampering-detection`.
It guarantees:
- Baseline fitting parameters strictly use the training split only.
- Evaluation metrics (accuracy, class distribution) are computed strictly on held-out test data.
- Clear distinction that synthetic fixture evaluations serve pipeline validation, not real-world video forensics benchmarks.

## Synthetic Fixture
An explicit 8-row synthetic classification fixture is provided in `fixture.json`:
- `x1 = [0, 1, 2, 3, 4, 5, 6, 7]`
- `x2 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`
- `label = [0, 0, 0, 0, 1, 1, 1, 1]`
- Stable IDs: `rec_0` through `rec_7`, with groups `grp_0` and `grp_1`
- Split:
  - Train: `["rec_0", "rec_1", "rec_2", "rec_4", "rec_5"]` (3 of class 0, 2 of class 1; majority = 0)
  - Test: `["rec_3", "rec_6", "rec_7"]` (1 of class 0, 2 of class 1)

## Running Tests
Run the offline CPU-only unittest discovery:
```bash
python3 -m unittest discover -s "projects/detection/Video-tampering-detection/course-checks/r43-t023" -p "test_*.py"
```
