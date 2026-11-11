# R43-T013: Training-mean regression baseline

This offline check compares a constant regression baseline with a fixed
synthetic fixture. The baseline is the arithmetic mean of training targets
and does not use the feature columns.

## Fixture and results

The fixture contains four training records, one validation record, and one
test record. Targets follow `target = 2*x1 + 1`; `x2` is always 1. The
training targets are 1, 3, 5, and 7, so the fitted baseline is 4.

The baseline predicts 4 for both held-out records:

| Record | Split | Actual | Prediction | Absolute error |
| --- | --- | ---: | ---: | ---: |
| r5 | Validation | 9 | 4 | 5 |
| r6 | Test | 11 | 4 | 7 |

## Run the tests

From the repository root:

```bash
python3 -m unittest discover -s 'projects/algorithms/Linear Regression Model/course-checks/r43-t013' -p 'test_*.py'

```

The tests verify training-only fitting, held-out predictions, and invalid training inputs. This synthetic fixture demonstrates a simple baseline; it does not establish real-world model performance.
