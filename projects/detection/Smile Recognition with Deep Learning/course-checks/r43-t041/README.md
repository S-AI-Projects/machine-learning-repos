# Schema Validation: Smile Recognition with Deep Learning (R43-T041)

## Overview
This component implements an explicit input schema validator for classification data fields in the Smile Recognition pipeline:
- `record_id`: Unique non-empty identifier.
- `group_id`: Group / subject identifier.
- `x1`: Finite numeric feature.
- `x2`: Finite numeric feature.
- `label`: Binary classification target in `{0, 1}`.

## Synthetic Fixture
The verification uses an 8-row deterministic synthetic fixture (`x1=[0..7], x2=1, label=[0,0,0,0,1,1,1,1]`) with stable record IDs `r1..r8` and group IDs `g1..g4`.
> **Note**: This data is strictly synthetic for pipeline interface verification and does not constitute real-world clinical/biometric performance claims.

## Running Tests
Execute the unit test suite:
```bash
python3 -m unittest discover -s 'projects/detection/Smile Recognition with Deep Learning/course-checks/r43-t041' -p "test_*.py"
```
