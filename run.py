#!/usr/bin/env python3
"""Run local YOLOX workflows with the current Python, without sourcing env.sh."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
COMMANDS = {
    "check-env": ["scripts/check_existing_env.py"],
    "build": ["scripts/build_yolox.py"],
    "sample-data": ["-m", "offline_yolox.sample_data"],
    "train": ["-m", "offline_yolox.train"],
    "preview": ["-m", "offline_yolox.preview"],
    "evaluate": ["-m", "offline_yolox.evaluate"],
    "validate-gpu": ["scripts/validate_gpu.py"],
}


def runtime_environment():
    """Configure child processes before they import NumPy, Torch or YOLOX."""
    environment = os.environ.copy()
    paths = [str(ROOT / "src"), str(ROOT / "third_party" / "YOLOX")]
    if environment.get("PYTHONPATH"):
        paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    # Direct invocation always uses the calling Python, including its venv.
    environment["YOLOX_PYTHON"] = sys.executable
    environment["PATH"] = os.pathsep.join(
        [str(Path(sys.executable).parent), environment.get("PATH", os.defpath)]
    )
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MAX_JOBS"):
        environment.setdefault(name, "2")
    environment.setdefault("TORCH_EXTENSIONS_DIR", str(ROOT / "artifacts" / "torch_extensions"))
    return environment


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Example: python run.py train --mode online --device cuda. "
               "Use COMMAND --help for command options. Relative paths start at the project root.",
    )
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Options for the selected command")
    args = parser.parse_args(argv)
    command = [sys.executable, *COMMANDS[args.command], *args.args]
    # Help can be read without libseccomp. Actual execution retains the existing
    # offline policy, inherited by DataLoader workers and compiler subprocesses.
    if not any(option in args.args for option in ("-h", "--help")):
        command = [sys.executable, str(ROOT / "scripts" / "no_network.py"), *command]
    try:
        result = subprocess.run(command, cwd=ROOT, env=runtime_environment())
    except KeyboardInterrupt:
        return 130
    return result.returncode if result.returncode >= 0 else 128 - result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
