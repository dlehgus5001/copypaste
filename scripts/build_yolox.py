#!/usr/bin/env python3
"""Build the bundled official FastCOCOEval extension using this Python."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "third_party" / "YOLOX"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    environment = os.environ.copy()
    for name, default in (("CC", "gcc"), ("CXX", "g++"), ("MAX_JOBS", "2")):
        environment.setdefault(name, default)
    # Use the same interpreter's build tools, without pip or build isolation.
    environment["PATH"] = os.pathsep.join(
        [str(Path(sys.executable).parent), environment.get("PATH", os.defpath)]
    )
    environment["PIP_NO_INDEX"] = "1"
    environment.setdefault("TORCH_EXTENSIONS_DIR", str(ROOT / "artifacts" / "torch_extensions"))
    environment["PYTHONPATH"] = str(VENDOR) + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    for name in ("CC", "CXX"):
        # Compiler variables may include flags (e.g. 'ccache g++'); setup.py
        # handles those itself. Diagnose simple missing executable names early.
        compiler = environment[name]
        if not any(character.isspace() for character in compiler) and not shutil.which(
            compiler, path=environment["PATH"]
        ):
            parser.error(f"Compiler not found: {compiler}. Install the OS C/C++ build tools.")
    steps = [
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        [sys.executable, "-c", "from yolox.layers import FastCOCOEvalOp; "
         "module = FastCOCOEvalOp().load(); "
         "assert hasattr(module, 'COCOevalEvaluateImages'); "
         "assert hasattr(module, 'COCOevalAccumulate'); "
         "print('Official FastCOCOEval C++ extension loaded:', module.__file__)"],
    ]
    for command in steps:
        completed = subprocess.run(command, cwd=VENDOR, env=environment)
        if completed.returncode:
            return completed.returncode if completed.returncode >= 0 else 128 - completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
