#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3.12}"
venv_dir="${1:-$project_dir/.venv}"
"$python_bin" -c 'import platform,sys; assert sys.version_info[:2]==(3,12), "Python 3.12 required"; assert platform.system()=="Linux" and platform.machine()=="x86_64", "Linux x86_64 required"'
"$python_bin" - "$project_dir/wheelhouse-manifest.json" <<'PY'
import json
import platform
import sys
target = json.load(open(sys.argv[1], encoding="utf-8"))["target"]
libc, version = platform.libc_ver()
required = tuple(map(int, target["minimum_glibc"].split(".")))
if libc != "glibc" or tuple(map(int, version.split("."))) < required:
    raise SystemExit(f"This bundle needs glibc >= {target['minimum_glibc']}; found {libc} {version}")
PY
"$python_bin" "$project_dir/scripts/verify_bundle.py" --wheels-only
if [[ ! -e "$venv_dir" ]]; then
  "$python_bin" -m venv "$venv_dir"
elif [[ ! -x "$venv_dir/bin/python" ]]; then
  echo "Refusing to overwrite an existing non-venv directory: $venv_dir" >&2
  exit 1
fi
"$venv_dir/bin/python" -c 'import sys; assert sys.version_info[:2]==(3,12), "Existing venv must use Python 3.12"'
"$venv_dir/bin/python" -m pip --isolated install \
  --no-index --find-links "$project_dir/wheelhouse" --require-hashes \
  --only-binary=:all: --no-cache-dir --disable-pip-version-check \
  -r "$project_dir/requirements-lock.txt"
"$venv_dir/bin/python" -m pip --isolated --no-cache-dir --disable-pip-version-check check
# Import upstream directly; do not invoke upstream setup.py or fetch its optional exporters.
"$venv_dir/bin/python" - "$project_dir" <<'PY'
from pathlib import Path
import site
import sys
root = Path(sys.argv[1])
upstream = root / "third_party" / "YOLOX"
if not (upstream / "yolox" / "__init__.py").is_file():
    raise SystemExit("Missing vendored YOLOX source")
Path(site.getsitepackages()[0], "yolox_offline.pth").write_text(
    str((root / "src").resolve()) + "\n" + str(upstream.resolve()) + "\n", encoding="utf-8")
PY
"$venv_dir/bin/python" "$project_dir/scripts/verify_bundle.py" --installed
echo "Offline environment ready: $venv_dir"
