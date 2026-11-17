# Course Check R43-T026: Bounded CPU Smoke Run

## Overview
This component implements a lightweight, deterministic CPU classification smoke run under `projects/algorithms/Transfer Learning with VGG16 and ResNet`.
It guarantees:
- Completely offline, CPU-only execution requiring zero GPU, zero network downloads, and zero credentials.
- Bounded training execution: processed on at most 32 samples for at most 1 epoch / $\le 20$ update steps.
- Verification of finite outputs/loss, valid prediction shapes, and recorded step metrics.
- Explicit labelling as a smoke verification test for interface correctness rather than real-world deep transfer learning claims.

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
python3 -m unittest discover -s "projects/algorithms/Transfer Learning with VGG16 and ResNet/course-checks/r43-t026" -p "test_*.py"
```
