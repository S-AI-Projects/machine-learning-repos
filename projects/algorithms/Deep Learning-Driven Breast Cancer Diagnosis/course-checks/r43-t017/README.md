# R43-T017: Reproducible synthetic classification configuration

This offline check records a JSON-serializable schema version, seed, ordered
features, fixed split membership, and threshold algorithm settings. The eight
rows are synthetic pipeline fixtures, not observations from the named breast
cancer dataset. Distinct stable group IDs keep the fixed stratified splits
separate.

Expected predictions are r1-r4: 0 and r5-r8: 1. Reloading the JSON config
reconstructs the exact split IDs and predictions. Unknown fields, incompatible
feature ordering, invalid split partitions, and contradictory labels are rejected.

Run from the repository root:

```bash
python3 -m unittest discover -s 'projects/algorithms/Deep Learning-Driven Breast Cancer Diagnosis/course-checks/r43-t017' -p 'test_*.py' -v
```

This is pipeline verification only, not evidence of real-world or clinical
performance. The original project script uses external imaging data and deep
learning libraries; this bounded check uses only the Python standard library.
