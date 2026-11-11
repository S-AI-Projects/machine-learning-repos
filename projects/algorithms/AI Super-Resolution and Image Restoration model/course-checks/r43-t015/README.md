# R43-T015: Explicit image preprocessing

This offline CPU check uses six synthetic 4×4 grayscale images: three all-zero images labeled 0 and three white-diagonal images labeled 1. Each record has a unique ID and declared group. The fixed fixture assigns one group to each train, validation, and test split; each split contains one record per class. No real dataset or model training is used.

`transform_pixels` accepts only a 4×4 grayscale image with finite numeric pixels in the inclusive range 0..255. It returns float32 values scaled to 0..1 with shape `(4, 4, 1)`. The transform is stateless and has no fit interface. `validate_splits` rejects duplicate record IDs and groups crossing split boundaries. This verifies the preprocessing contract only, not real-world or clinical performance.

## Run

From the repository root:

```bash
python3 -m unittest discover -s "projects/algorithms/AI Super-Resolution and Image Restoration model/course-checks/r43-t015" -p "test_*.py" -v
```
