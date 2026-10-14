#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$PROJECT_ROOT"
exec python scripts/no_network.py python -m offline_yolox.evaluate "$@"
