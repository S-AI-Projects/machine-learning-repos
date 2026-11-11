# R43-T035: Make preprocessing explicit

This isolated, offline CPU check uses **synthetic pipeline-verification data**. Its rows are not observations from the named project dataset. No score here measures real-world, medical, financial, or clinical performance.

## Contract and split

record_id: nonempty string; group_id: nonempty string; x1/x2: finite numeric values; label: integer 0 or 1. Unknown or malformed input is rejected unless a task below states a different policy.

The fixture contains stable record IDs `r0` onward and distinct group IDs `g0` onward. Seed: 43. The fixed split is `train=['r0', 'r1', 'r4', 'r5']`, `validation=['r2', 'r6']`, `test=['r3', 'r7']`. Each split includes both classes. No group crosses a split; validation and test rows never fit parameters. For a real-data extension, separately record source, licence, access requirements, and group/time-aware split design.

Run `python3 -m unittest discover -s 'projects/detection/Fashion Detector/course-checks/r43-t035' -p "test_*.py"` from the repository root. Only Python 3 standard library is required.

## Result and limits

Fit uses only train IDs and rejects overlapping or incomplete split coverage. It validates the binary label set and rejects missing/nonfinite features. Transform accepts fixed feature order `(x1,x2)`, returns rows of double-precision floats, and cannot refit. Training means are `(2.5,1.0)`; zero-variance `x2` uses scale 1 and transforms to 0.
