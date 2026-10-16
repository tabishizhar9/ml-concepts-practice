#!/usr/bin/env python3
from pathlib import Path
import os
import re
import subprocess
import sys


def main():
    project_root = Path(__file__).resolve().parent
    project_path = project_root / "code"

    # Make sure the project folder is on PYTHONPATH
    pythonpath = os.environ.get("PYTHONPATH", "")
    parts = [p for p in pythonpath.split(os.pathsep) if p]
    if str(project_path) not in parts:
        parts.append(str(project_path))
        os.environ["PYTHONPATH"] = os.pathsep.join(parts)

    os.chdir(project_path)

    print()
    print("Starting Assignment 1 Tests...")
    print("=" * 70)

    total_score = 0.0
    bonus_score = 0.0

    test_files = [
        "tests/test_numpy_basics.py",
        "tests/test_numpy_applied.py",
        "tests/test_torch_tensors_basics.py",
        "tests/test_derivatives_and_graphs.py",
        "tests/test_linear_regression.py",
        "tests/test_logistic_regression.py",
    ]

    for rel_test in test_files:
        test_file_name = Path(rel_test).name
        notebook_name = f"{test_file_name.replace('test_', '')}.ipynb"

        print()
        print()
        print(f"RUNNING {test_file_name}...")
        print()

        result = subprocess.run(
            [sys.executable, "-m", "unittest", rel_test],
            cwd=project_path,
            capture_output=True,
            text=True,
        )

        output = result.stdout + result.stderr
        sys.stdout.write(output)
        if result.stderr:
            sys.stderr.write(result.stderr)

        if result.returncode != 0:
            print(f"There was an error running the tests in {test_file_name}. Check the test script.")
            print("-" * 70)
            continue

        notebook_score = 0.0
        notebook_bonus_score = 0.0

        for line in output.splitlines():
            if "(" in line:
                continue

            if "Score achieved:" in line:
                match = re.search(r"Score achieved:\s*\[([0-9]+(?:\.[0-9]+)?)\s*/", line)
                if match:
                    notebook_score += float(match.group(1))

            elif "Bonus score achieved:" in line:
                match = re.search(r"Bonus score achieved:\s*\[([0-9]+(?:\.[0-9]+)?)\s*/", line)
                if match:
                    notebook_bonus_score += float(match.group(1))

        print()
        print(f"TOTAL SCORE FOR {notebook_name}: [ {notebook_score} ]")
        print(f"BONUS SCORE: [ {notebook_bonus_score} ]")
        print("-" * 70)

        total_score += notebook_score
        bonus_score += notebook_bonus_score

    print()
    print()
    print("=" * 70)
    print(f"TOTAL SCORE FOR **Assignment 1 Coding part**: [ {total_score} / 60.0 ]")
    print(f"BONUS SCORE: [ {bonus_score} / 5.0 ]")
    print("=" * 70)


if __name__ == "__main__":
    main()