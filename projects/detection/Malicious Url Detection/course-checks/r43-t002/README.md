# R43-T002 — Leakage-Free Split Manifest: Malicious URL Detection

**Task type:** Leakage-free splitting
**Student:** RUHAN JANGIR (E26B07F0583)
**Branch:** `student/e26b07f0583/R43-T002`

> **⚠️ Synthetic Fixture Notice**
> All records are **synthetic fixtures for pipeline verification only**.
> NOT drawn from any real malicious-URL dataset.

## Fixture (8 synthetic rows)

| record_id | group_id | x1 | x2 | label |
|-----------|----------|----|----|-------|
| s0–s3 | g0–g3 | 0–3 | 1 | 0 |
| s4–s7 | g4–g7 | 4–7 | 1 | 1 |

## Deterministic Split (seed=42, stratified)

| Split | Record IDs | Count |
|-------|-----------|-------|
| train | s0,s1,s2,s4,s5 | 5 |
| validation | s3 | 1 |
| test | s6,s7 | 2 |

## Guarantees

- Splits are disjoint (no ID in more than one split)
- Union of splits = all 8 records
- Repeated calls return identical manifests
- Overlapping or unknown IDs raise `ValueError`

## Running Checks

```bash
python3 -m unittest discover -s 'projects/detection/Malicious Url Detection/course-checks/r43-t002' -p "test_*.py"
```

All tests offline, CPU-only, standard library only.
