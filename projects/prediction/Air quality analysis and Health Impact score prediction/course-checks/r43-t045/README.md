# Explicit Preprocessing Contract: Air Quality & Health Impact (R43-T045)

## Overview
Implements an explicit numeric preprocessing contract:
- Scaling parameters (min-max) are fitted exclusively on training rows.
- Validates feature ordering (`['x1', 'x2']`), label set (`{0, 1}`), and strict missing-value rejection policy.
- Asserts that fitting on held-out input is structurally impossible via the `transform()` interface.
- Rejects deliberate contamination fixtures (missing attributes, non-numeric values, out-of-range labels).

## Fixture Specification
Synthetic 8-row classification fixture (`x1=[0..7], x2=1, label=[0,0,0,0,1,1,1,1]`) with stratified train/test split.
> **Verification Boundary**: Labeled strictly as synthetic pipeline verification, not real environmental or health claims.

## Running Tests
```bash
python3 -m unittest discover -s 'projects/prediction/Air quality analysis and Health Impact score prediction/course-checks/r43-t045' -p "test_*.py"
```
