# Course Check R43-T027: Reproducible Configuration

## Overview
This component implements a serializable, validated configuration manager under `projects/classification/Ecommerce product categorization`.
It guarantees:
- Complete serializability of random seed, schema version, feature ordering, split IDs, and algorithm settings.
- Deterministic reconstruction of split IDs and predictions upon reloading.
- Strict rejection of unknown configuration fields, contradictory split partitions, and incompatible input shapes.
- Recording of runtime environment versions (Python, platform).

## Synthetic Fixture
An explicit 8-row synthetic classification fixture is provided in `fixture.json`:
- `x1 = [0, 1, 2, 3, 4, 5, 6, 7]`
- `x2 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`
- `label = [0, 0, 0, 0, 1, 1, 1, 1]`
- Stable IDs: `rec_0` through `rec_7`, with groups `grp_0` and `grp_1`
- Split:
  - Train: `["rec_0", "rec_1", "rec_4", "rec_5"]`
  - Test: `["rec_2", "rec_3", "rec_6", "rec_7"]`

## Running Tests
Run the offline CPU-only unittest discovery:
```bash
python3 -m unittest discover -s "projects/classification/Ecommerce product categorization/course-checks/r43-t027" -p "test_*.py"
```
