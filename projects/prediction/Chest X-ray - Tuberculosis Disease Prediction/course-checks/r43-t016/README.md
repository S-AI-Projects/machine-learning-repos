# R43-T016: Bounded CPU classification smoke run

This offline check uses eight synthetic rows with x1 values 0 through 7,
x2 = 1, and labels [0, 0, 0, 0, 1, 1, 1, 1]. Rows have stable record and
group IDs. They are not observations from the named chest X-ray dataset.

A deterministic split with seed 43 stratifies each class into train,
validation, and test. A standard-library logistic baseline trains on the four
training rows for one epoch and four update steps. The run is limited to 32
samples and 20 updates. Tests check finite loss and predictions, output shapes,
split integrity, and invalid inputs. This is interface smoke testing only; it
does not demonstrate useful accuracy or clinical performance.

The check uses only the Python standard library and requires no GPU, dataset,
credentials, or network access.

## Run

From the repository root:

    python3 -m unittest discover -s 'projects/prediction/Chest X-ray - Tuberculosis Disease Prediction/course-checks/r43-t016' -p 'test_*.py' -v
