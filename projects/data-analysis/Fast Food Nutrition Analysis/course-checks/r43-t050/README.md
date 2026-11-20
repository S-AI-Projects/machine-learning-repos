# Saved Pipeline Interface: Fast Food Nutrition Analysis (R43-T050)

## Overview
Implements a secure saved-pipeline interface for Fast Food Nutrition Analysis classification:
- Serializes trained parameters (`weights`, `bias`, `threshold`, `interface_version`) strictly to standard JSON (avoiding unsafe pickle deserialization).
- Reloads parameters and exposes schema-validated inference requiring zero training data or credentials.
- Asserts before/after prediction agreement within exact tolerance (`1e-6`).
- Validates version compatibility and input schema, failing clearly on malformed records.

## Fixture Specification
Synthetic 8-row classification fixture (`x1=[0..7], x2=1, label=[0,0,0,0,1,1,1,1]`) with stratified train/test split.
> **Verification Boundary**: Labeled strictly as synthetic pipeline verification, not real nutritional or health analysis.

## Running Tests
```bash
python3 -m unittest discover -s 'projects/data-analysis/Fast Food Nutrition Analysis/course-checks/r43-t050' -p "test_*.py"
```
