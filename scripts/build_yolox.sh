#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${YOLOX_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
export MAX_JOBS="${MAX_JOBS:-2}"
exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/build_yolox.py" "$@"
