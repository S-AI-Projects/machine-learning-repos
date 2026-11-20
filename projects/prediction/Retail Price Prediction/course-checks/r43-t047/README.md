# Reproducible Configuration: Retail Price Prediction (R43-T047)

## Overview
Implements a strict serializable configuration schema for reproducible training and inference:
- Serializes `schema_version`, `seed`, `feature_ordering`, `split_ids`, `algorithm_settings`, and `environment_metadata`.
- Strict validation: rejects unknown configuration keys, contradictory settings (e.g. split overlaps, invalid seeds), and unknown algorithms.
- Enforces input shape validation at prediction time.
- Guarantees exact reconstruction of splits and predictions from serialized JSON.

## Fixture Specification
Synthetic 8-row classification fixture (`x1=[0..7], x2=1, label=[0,0,0,0,1,1,1,1]`) with stratified train/test split.
> **Verification Boundary**: Labeled strictly as synthetic pipeline verification, not real retail pricing or commercial optimization.

## Running Tests
```bash
python3 -m unittest discover -s 'projects/prediction/Retail Price Prediction/course-checks/r43-t047' -p "test_*.py"
```
