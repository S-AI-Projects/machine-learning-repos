# Minimal Baseline: Data Analysis of Tech Stocks (R43-T043)

## Overview
Implements the simplest honest baseline for binary classification:
- Predicts majority class computed strictly from training split records.
- Evaluates accuracy on held-out test records.
- Guarantees parameter isolation: test split data never affects baseline parameters.

## Fixture Specification
Synthetic 8-row classification fixture (`x1=[0..7], x2=1, label=[0,0,0,0,1,1,1,1]`) with stratified train/test split.
> **Verification Notice**: Synthetic fixture for pipeline contract verification only; no real-world market or stock prediction performance claimed.

## Running Tests
```bash
python3 -m unittest discover -s 'projects/data-analysis/Data Analysis of Tech Stocks/course-checks/r43-t043' -p "test_*.py"
```
