# Error Analysis Report: Stock Price Prediction (R43-T048)

## Overview
Implements an error-analysis reporting pipeline for classification monitoring:
- Generates a structured error report with `record_id`, `expected`, `predicted`, and a classified `reason_category`.
- Verified with one injected wrong prediction (`record_id: r5`, expected 1, predicted 0) to ensure testability.
- Strictly asserts that correct predictions do not appear in the report and that aggregate error counts match detailed rows.

## Fixture Specification
Synthetic 8-row classification fixture (`x1=[0..7], x2=1, label=[0,0,0,0,1,1,1,1]`) with stratified train/test split.
> **Verification Boundary**: Labeled strictly as synthetic pipeline verification, not real financial or stock price prediction.

## Running Tests
```bash
python3 -m unittest discover -s projects/prediction/Stock_Price_Prediction/course-checks/r43-t048 -p "test_*.py"
```
