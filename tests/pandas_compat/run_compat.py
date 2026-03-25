#!/usr/bin/env python3
"""Run pandas compatibility tests and report results."""
import sys
import subprocess

# Run pytest on the compat directory
result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/pandas_compat/", "-v", "--tb=short"] + sys.argv[1:],
    check=False,
)
sys.exit(result.returncode)
