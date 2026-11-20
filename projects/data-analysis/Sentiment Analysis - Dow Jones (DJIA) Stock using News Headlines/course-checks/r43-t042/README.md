# Deterministic Split Manifest: Sentiment Analysis - DJIA News Headlines (R43-T042)

## Overview
Implements deterministic, leakage-free splitting with strict chronological ordering for text time-series data.
- Ensures `train`, `val`, and `test` splits are strictly disjoint.
- Enforces time isolation: `train` timestamps <= `val` timestamps <= `test` timestamps.
- Raises explicit `SplitLeakageError` on overlapping record IDs or time inversions.

## Fixture Specification
Synthetic text fixture containing 6 fictional records (`t1..t6`) with labels `[1, 1, 0, 0, 1, 0]`.
> **Verification Boundary**: Labeled strictly as synthetic pipeline verification, not real financial or sentiment predictions.

## Running Tests
```bash
python3 -m unittest discover -s 'projects/data-analysis/Sentiment Analysis - Dow Jones (DJIA) Stock using News Headlines/course-checks/r43-t042' -p "test_*.py"
```
