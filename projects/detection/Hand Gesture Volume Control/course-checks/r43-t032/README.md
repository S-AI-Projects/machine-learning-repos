# R43-T032: Prevent split leakage

This isolated, offline CPU check uses **synthetic pipeline-verification data**. Its rows are not observations from the named project dataset. No score here measures real-world, medical, financial, or clinical performance.

## Contract and split

record_id: nonempty string; group_id: nonempty string; x1/x2: finite numeric values; label: integer 0 or 1. Unknown or malformed input is rejected unless a task below states a different policy.

The fixture contains stable record IDs `r0` onward and distinct group IDs `g0` onward. Seed: 43. The fixed split is `train=['r0', 'r1', 'r4', 'r5']`, `validation=['r2', 'r6']`, `test=['r3', 'r7']`. Each split includes both classes. No group crosses a split; validation and test rows never fit parameters. For a real-data extension, separately record source, licence, access requirements, and group/time-aware split design.

Run `python3 -m unittest discover -s 'projects/detection/Hand Gesture Volume Control/course-checks/r43-t032' -p "test_*.py"` from the repository root. Only Python 3 standard library is required.

## Result and limits

The manifest groups rows before splitting. Without time it sorts group IDs within each class and assigns the final two groups of each class to validation and test. With a `time` field on every row it uses whole groups chronologically. It rejects record overlap, group leakage, partial timestamps, and overlapping group time ranges. The fixture manifest exactly matches `fixture.json`.
