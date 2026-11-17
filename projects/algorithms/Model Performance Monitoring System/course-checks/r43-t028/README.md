# Course Check R43-T028: Error Analysis Report

## Overview
This component implements an automated error-analysis reporting tool under `projects/algorithms/Model Performance Monitoring System`.
It guarantees:
- Granular error capture: logs stable record ID, expected ground truth, predicted value, and reason category.
- Accurate isolation: correctly predicted records are excluded from the error table.
- Exact match between aggregate error statistics and detailed row-level findings.
- Testability: verified using a deliberately injected false negative misclassification on `rec_6`.

## Synthetic Fixture
An explicit 8-row synthetic classification fixture is provided in `fixture.json`:
- `x1 = [0, 1, 2, 3, 4, 5, 6, 7]`
- `x2 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`
- `label = [0, 0, 0, 0, 1, 1, 1, 1]`
- `injected_prediction = [0, 0, 0, 0, 1, 1, 0, 1]` (injected error on `rec_6`)
- Stable IDs: `rec_0` through `rec_7`, with groups `grp_0` and `grp_1`

## Running Tests
Run the offline CPU-only unittest discovery:
```bash
python3 -m unittest discover -s "projects/algorithms/Model Performance Monitoring System/course-checks/r43-t028" -p "test_*.py"
```
