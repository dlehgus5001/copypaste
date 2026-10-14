#!/usr/bin/env bash
# Source this file to use the bundled YOLOX source without online installation.
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/third_party/YOLOX${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$PROJECT_ROOT/.venv/bin:$PATH"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export MAX_JOBS="${MAX_JOBS:-2}"
export TORCH_EXTENSIONS_DIR="$PROJECT_ROOT/artifacts/torch_extensions"
