#!/usr/bin/env bash
# Source this file to use the bundled YOLOX source without online installation.
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${YOLOX_PYTHON:-}" ]]; then
  selected_python="$(command -v -- "$YOLOX_PYTHON")" || {
    echo "Cannot find selected YOLOX_PYTHON: $YOLOX_PYTHON" >&2
    return 1
  }
else
  selected_python="$PROJECT_ROOT/.venv/bin/python"
fi
if [[ ! -x "$selected_python" ]]; then
  echo "Python interpreter missing: $selected_python. Install the venv or set YOLOX_PYTHON to the existing environment's Python." >&2
  return 1
fi
selected_python="$(cd -- "$(dirname -- "$selected_python")" && pwd)/$(basename -- "$selected_python")"
"$selected_python" -c 'import sys; assert sys.version_info[:2] == (3, 12), "This profile requires Python 3.12"' || return 1
export YOLOX_PYTHON="$selected_python"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/third_party/YOLOX${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$(dirname -- "$YOLOX_PYTHON"):$PATH"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export MAX_JOBS="${MAX_JOBS:-2}"
export TORCH_EXTENSIONS_DIR="$PROJECT_ROOT/artifacts/torch_extensions"
