# Metric Correctness: World Population Growth Analysis (R43-T044)

## Overview
Independently implements and verifies standard evaluation metrics with explicit zero-denominator behavior:
- Classification precision, recall, accuracy, and F1.
- Exact test verification: `truth=[1,1,0,0]` with `prediction=[1,0,1,0]` produces `precision=0.5` and `recall=0.5`.
- Cross-check regression MAE: `truth=[1,2,3]` with `prediction=[1,3,2]` produces `MAE=2/3`.
- Explicit handling of zero denominators via `zero_division` parameter.

## Synthetic Fixture
Synthetic 8-row classification fixture (`x1=[0..7], x2=1, label=[0,0,0,0,1,1,1,1]`) with stratified train/test split.
> **Verification Boundary**: Labeled strictly as synthetic pipeline verification, not real demographic or population predictions.

## Running Tests
```bash
python3 -m unittest discover -s 'projects/data-analysis/World Population Growth Analysis/course-checks/r43-t044' -p "test_*.py"
```
