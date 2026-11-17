# R43-T004 — Metric Correctness: Parkinson Disease Detection

**Task type:** Metric correctness | **Student:** RUHAN JANGIR (E26B07F0583)

> ⚠️ Synthetic Fixture. NOT clinical Parkinson disease data.

## Spec Example
`truth=[1,1,0,0] / pred=[1,0,1,0]` → precision=recall=0.5

Zero-denominator policy: returns 0.0 (explicit).

## Running Checks
```bash
python3 -m unittest discover -s 'projects/detection/Parkinson Disease Detection/course-checks/r43-t004' -p "test_*.py"
```
