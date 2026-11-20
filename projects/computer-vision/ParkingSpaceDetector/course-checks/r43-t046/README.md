# Bounded CPU Smoke Run: ParkingSpaceDetector (R43-T046)

## Overview
Implements a tiny CPU smoke run for classification pipeline verification:
- Bounded sample size: <= 32 samples (test fixture uses 8 samples).
- Single epoch / <= 20 update steps.
- CPU-only execution: requires no GPU, API keys, credentials, or remote data downloads.
- Asserts interface correctness, finite loss values, and valid prediction output shapes.

## Fixture Specification
Synthetic 8-row classification fixture (`x1=[0..7], x2=1, label=[0,0,0,0,1,1,1,1]`) with stratified train/test split.
> **Verification Boundary**: Labeled strictly as synthetic pipeline verification, not real computer-vision parking occupancy performance.

## Running Tests
```bash
python3 -m unittest discover -s projects/computer-vision/ParkingSpaceDetector/course-checks/r43-t046 -p "test_*.py"
```
