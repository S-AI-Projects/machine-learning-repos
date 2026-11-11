# R43-T036: Add a bounded CPU smoke run

This isolated, offline CPU check uses **synthetic pipeline-verification data**. Its rows are not observations from the named project dataset. No score here measures real-world, medical, financial, or clinical performance.

## Contract and split

record_id: nonempty string; group_id: nonempty string; pixels: 4×4 grayscale integers 0..255; label: integer 0 or 1. Unknown or malformed input is rejected unless a task below states a different policy.

The fixture contains stable record IDs `r0` onward and distinct group IDs `g0` onward. Seed: 43. The fixed split is `train=['r0', 'r3']`, `validation=['r1', 'r4']`, `test=['r2', 'r5']`. Each split includes both classes. No group crosses a split; validation and test rows never fit parameters. For a real-data extension, separately record source, licence, access requirements, and group/time-aware split design.

Run `python3 -m unittest discover -s 'projects/prediction/Digit Prediction using cnn/course-checks/r43-t036' -p "test_*.py"` from the repository root. Only Python 3 standard library is required.

## Result and limits

This is an explicitly naive, untrained diagonal detector, not a CNN. It checks six 4×4 grayscale images, uses no GPU or download, records zero training steps/epochs, returns six predictions and finite mean-squared loss. The fixed fixture loss is 0; this is an interface smoke result only.
