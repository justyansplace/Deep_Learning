#!/usr/bin/env bash
set -euo pipefail
python -m compileall hw tests_public
pytest -q tests_public
