# R43-T020: Saved pipeline interface

This offline teaching check stores a small, fixed two-feature classifier's
configuration and parameters as JSON, reloads them, validates the saved schema
version, and exposes inference without training data. JSON is used instead of
pickle so loading does not execute serialized code.

The eight rows are synthetic: x1 is 0 through 7, x2 is always 1, and labels
are 0 for x1 below 4 and 1 otherwise. Stable record and group IDs are provided.
The fixed seed-20 split is stratified: train r1/r3/r5/r7, validation
r2/r6, and test r4/r8. These are not observations from the named river-water
dataset and do not demonstrate real-world water-quality performance.

The saved parameters implement a deterministic logistic-shaped demonstration
interface, not a model trained on the named dataset. Before/after probabilities
are compared with absolute tolerance 1e-12. Input and parameter schema/version
mismatches fail clearly. Inference needs only JSON parameters and feature rows.

Run from the repository root:

```bash
python3 -m unittest discover -s 'projects/prediction/Ammonium Prediction In River Water/course-checks/r43-t020' -p 'test_*.py' -v
```

The check uses only Python's standard library, CPU, and local synthetic data.
