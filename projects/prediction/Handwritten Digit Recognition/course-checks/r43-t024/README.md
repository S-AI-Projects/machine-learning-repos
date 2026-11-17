# Course Check R43-T024: Evaluation Metric Verification

## Overview
This component verifies classification and regression evaluation metric contracts under `projects/prediction/Handwritten Digit Recognition`.
It strictly cross-checks:
- Binary classification precision, recall, accuracy, and F1 calculations.
- Exact verification of the canonical case: `truth=[1,1,0,0]` / `prediction=[1,0,1,0]` $\rightarrow$ `precision = recall = 0.5`.
- Exact regression cross-check: `truth=[1,2,3]` / `prediction=[1,3,2]` $\rightarrow$ $\text{MAE} = \frac{2}{3}$.
- Explicit zero-denominator behavior handling (when $TP + FP = 0$ or $TP + FN = 0$).

## Synthetic Fixture
A 6-record synthetic $4 \times 4$ grayscale image fixture is provided in `fixture.json`:
- 3 all-zero images with label 0 (`img_0`, `img_1`, `img_2`), group `grp_zero`.
- 3 diagonal white images (intensity 255 along main diagonal) with label 1 (`img_3`, `img_4`, `img_5`), group `grp_diag`.

*Notice: This fixture is an offline CPU-only smoke fixture for evaluation metric verification, not observations from real-world handwritten digit datasets.*

## Running Tests
Run the offline CPU-only unittest discovery:
```bash
python3 -m unittest discover -s "projects/prediction/Handwritten Digit Recognition/course-checks/r43-t024" -p "test_*.py"
```
