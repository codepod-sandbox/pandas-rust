#!/usr/bin/env bash
set -euo pipefail

BINARY="${1:?Usage: run_tests.sh <pandas-python-binary>}"
exec "$BINARY" -m pytest tests/python/ -v
