import os
import subprocess
import sys

def main():
    test_dir = os.path.join("projects", "data-analysis", "Fast Food Nutrition Analysis", "course-checks", "r43-t050")
    cmd = [sys.executable, "-m", "unittest", "discover", "-s", test_dir, "-p", "test_*.py"]
    res = subprocess.run(cmd)
    if res.returncode != 0:
        sys.exit(res.returncode)
    print("All R43-T050 saved pipeline interface checks passed successfully.")

if __name__ == "__main__":
    main()
