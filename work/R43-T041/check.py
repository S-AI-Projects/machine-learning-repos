import os
import subprocess
import sys

def main():
    test_dir = os.path.join("projects", "detection", "Smile Recognition with Deep Learning", "course-checks", "r43-t041")
    cmd = [sys.executable, "-m", "unittest", "discover", "-s", test_dir, "-p", "test_*.py"]
    res = subprocess.run(cmd)
    if res.returncode != 0:
        sys.exit(res.returncode)
    print("All R43-T041 validation checks passed successfully.")

if __name__ == "__main__":
    main()
