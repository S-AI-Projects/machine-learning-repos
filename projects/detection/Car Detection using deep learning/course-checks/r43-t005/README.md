# R43-T005 — Preprocessing Contract: Car Detection using Deep Learning

**Task type:** Preprocessing contract | **Student:** RUHAN JANGIR (E26B07F0583)

> ⚠️ 8 synthetic rows. NOT real car detection data.

## Contract
- Features: `x1`, `x2` (required, numeric)
- Labels: `{0, 1}` (unknown labels raise ValueError)
- Missing values: raise ValueError — no silent imputation
- Scaler fitted on training rows only

## Running Checks
```bash
python3 -m unittest discover -s 'projects/detection/Car Detection using deep learning/course-checks/r43-t005' -p "test_*.py"
```
