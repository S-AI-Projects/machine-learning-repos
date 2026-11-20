# Input Perturbation Robustness: Exoplanet Detection (R43-T049)

## Overview
Implements input perturbation testing for classification interface robustness:
- Asserts order-invariance under record permutation after stable ID alignment.
- Rejects unsupported inputs (missing `record_id`, missing or non-numeric feature fields).
- Supports explicit fallback mapping for graceful error recovery.
- Scoped strictly to interface contract verification without claiming broad adversarial robustness.

## Fixture Specification
Synthetic 8-row classification fixture (`x1=[0..7], x2=1, label=[0,0,0,0,1,1,1,1]`) with stratified train/test split.
> **Verification Boundary**: Labeled strictly as synthetic pipeline verification, not real astrophysical or exoplanetary detection claims.

## Running Tests
```bash
python3 -m unittest discover -s 'projects/detection/Exoplanet Detection/course-checks/r43-t049' -p "test_*.py"
```
