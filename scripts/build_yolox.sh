#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${YOLOX_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
export MAX_JOBS="${MAX_JOBS:-2}"
export PATH="$(dirname -- "$PYTHON_BIN"):$PATH"
# This Python distribution's sysconfig may reference an unavailable clang.
# Explicitly select installed compilers; caller-provided choices take priority.
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"
export PIP_NO_INDEX=1
cd "$PROJECT_ROOT/third_party/YOLOX"
# The official setup imports torch at build time; install the locked wheels first.
# No pip dependency resolution or isolated build environment is used here.
"$PYTHON_BIN" setup.py build_ext --inplace
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - <<'PY'
from yolox.layers import FastCOCOEvalOp
module = FastCOCOEvalOp().load()
assert hasattr(module, "COCOevalEvaluateImages")
assert hasattr(module, "COCOevalAccumulate")
print("Official FastCOCOEval C++ extension loaded:", module.__file__)
PY
