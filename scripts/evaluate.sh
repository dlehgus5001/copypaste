#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$PROJECT_ROOT"
exec "$YOLOX_PYTHON" scripts/no_network.py "$YOLOX_PYTHON" -m offline_yolox.evaluate "$@"
