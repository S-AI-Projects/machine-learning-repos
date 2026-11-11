# R43-T018: Synthetic error analysis

This offline example uses eight synthetic records with stable record and group
IDs. The fixed seed and stratified train, validation, and test split are recorded
in `fixture.json`; the rows are not observations from the named Reddit dataset.

The injected error is r6 (expected 1, predicted 0), categorized as
`threshold_boundary_case`. The report includes only incorrect predictions and
aggregates totals by split. No real text, personal data, model performance, or
clinical claim is involved.

Run from the repository root:

```bash
python3 -m unittest discover -s 'projects/detection/Sucide & Depression Detection/course-checks/r43-t018' -p 'test_*.py' -v
```

The bounded check uses only Python's standard library. It verifies a report
contract, not a deployable suicide or depression detector.
