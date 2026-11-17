# R43-T006 — CPU Smoke Run: Sign Language Detection

**Task type:** Small training/inference smoke test | **Student:** RUHAN JANGIR (E26B07F0583)

> ⚠️ Six fictional text records. NOT real sign language data. Pipeline fixture only.

## Constraints
- ≤ 32 samples, ≤ 1 epoch, ≤ 20 update steps
- CPU-only, no GPU, no credentials, no dataset download

## Running Checks
```bash
python3 -m unittest discover -s projects/detection/Sign_Language_Detection/course-checks/r43-t006 -p "test_*.py"
```
