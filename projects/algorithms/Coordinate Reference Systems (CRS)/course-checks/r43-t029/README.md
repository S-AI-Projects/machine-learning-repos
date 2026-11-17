# Course Check R43-T029: Input Perturbation Robustness

## Overview
This component implements an input perturbation test suite under `projects/algorithms/Coordinate Reference Systems (CRS)`.
It guarantees:
- Transformation invariance: Record-order permutation preserves documented prediction outputs when aligned by stable `record_id`.
- Rejection or explicit fallback: Missing or malformed features are strictly rejected or routed to a defined fallback mapping.
- Scope bounds: Explicitly asserts interface stability under valid transformations without claiming broad adversarial robustness.

## Synthetic Fixture
An explicit 8-row synthetic classification fixture is provided in `fixture.json`:
- `x1 = [0, 1, 2, 3, 4, 5, 6, 7]`
- `x2 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`
- `label = [0, 0, 0, 0, 1, 1, 1, 1]`
- Stable IDs: `rec_0` through `rec_7`, with groups `grp_0` and `grp_1`

## Running Tests
Run the offline CPU-only unittest discovery:
```bash
python3 -m unittest discover -s "projects/algorithms/Coordinate Reference Systems (CRS)/course-checks/r43-t029" -p "test_*.py"
```
