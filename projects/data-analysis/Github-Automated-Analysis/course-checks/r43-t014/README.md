# R43-T014: Verify classification metrics

This offline check independently verifies binary precision and recall. It uses
only the Python standard library and does not run the external-service
Streamlit application.

## Metric contract

The positive class is label `1`; averaging is binary, scoring only that class.
Precision is TP / (TP + FP), and recall is TP / (TP + FN). A zero denominator
returns `0.0`. For truth `[1, 1, 0, 0]` and predictions `[1, 0, 1, 0]`, TP=1,
FP=1, FN=1, so precision=0.5 and recall=0.5.

## Synthetic fixture and fixed split

The eight synthetic records use the exact fields `record_id`, `group_id`,
`x1`, `x2`, and `label`. They have x1 values 0 through 7, x2=1, and labels
`[0, 0, 0, 0, 1, 1, 1, 1]`. The fixed split is stratified by label:

- Train: row-001, row-003, row-005, row-007
- Validation: row-002, row-006
- Test: row-004, row-008

Each record and group ID is stable and unique; groups do not cross splits.
These rows are synthetic pipeline checks, not observations from a real
dataset and not evidence of real-world or clinical performance.

## Run

From the repository root:

```bash
python3 -m unittest discover -s 'projects/data-analysis/Github-Automated-Analysis/course-checks/r43-t014' -p 'test_*.py'
