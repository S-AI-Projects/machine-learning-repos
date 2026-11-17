# Course Check R43-T022: Split Leakage Prevention

## Overview
This component implements a deterministic split manifest generator and split-leakage detector under `projects/algorithms/Preprocessing Techniques of DL`.
It guarantees:
- Complete disjointness across train, validation, and test splits.
- Rejection of overlapping record IDs and group IDs across splits.
- Deterministic, reproducible partition manifests.
- Chronological preservation for time-series splits.

## Synthetic Fixture
An explicit 8-row synthetic classification fixture is provided in `fixture.json`:
- `x1 = [0, 1, 2, 3, 4, 5, 6, 7]`
- `x2 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`
- `label = [0, 0, 0, 0, 1, 1, 1, 1]`
- Stable IDs: `rec_0` through `rec_7`, with groups `grp_0` through `grp_3` and timestamps.

*Notice: This fixture is strictly for pipeline verification and split isolation testing, not real-world dataset observations.*

## Running Tests
Run the offline CPU-only unittest discovery:
```bash
python3 -m unittest discover -s "projects/algorithms/Preprocessing Techniques of DL/course-checks/r43-t022" -p "test_*.py"
```
