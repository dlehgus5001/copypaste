#!/usr/bin/env python3
"""Package source and, optionally, checksum-verified offline wheels.

Never includes venvs, credentials, user datasets, checkpoints or build outputs.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
ENTRIES = [".gitignore", "pyproject.toml", "README.md", "OFFLINE_INSTALL.md",
           "environment_report.md", "compatibility_report.md", "TEST_REPORT.md",
           "requirement.txt", "requirements.txt", "constraints-numpy126.txt",
           "constraints-existing-server.txt",
           "requirements-build.txt", "requirements-core.txt",
           "requirements-lock.txt", "wheelhouse-manifest.json",
           "src", "scripts", "configs", "tests", "reports", "third_party"]
SKIP_DIRS = {"__pycache__", ".pytest_cache", "build", ".git"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-wheels", action="store_true")
    parser.add_argument("--output", default="artifacts/yolox-source.tar.gz")
    args = parser.parse_args()
    subprocess.run([sys.executable, str(ROOT / "scripts/check_upstream.py")], check=True)
    if args.include_wheels:
        subprocess.run([sys.executable, str(ROOT / "scripts/verify_bundle.py"), "--wheels-only"], check=True)
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing archive: {output}")
    names = ENTRIES + (["wheelhouse"] if args.include_wheels else [])
    files = []
    for name in names:
        item = ROOT / name
        if not item.exists():
            raise SystemExit(f"Required bundle entry missing: {item}")
        candidates = item.rglob("*") if item.is_dir() else [item]
        for path in candidates:
            relative = path.relative_to(ROOT)
            if (path.is_file() and not path.is_symlink()
                    and not any(part in SKIP_DIRS or part.endswith(".egg-info") for part in relative.parts)
                    and path.suffix not in {".so", ".pyc", ".pyo", ".o"}):
                files.append(path)
    temporary = output.with_suffix(output.suffix + ".partial")
    try:
        with tarfile.open(temporary, mode="w:gz", compresslevel=1) as archive:
            for path in sorted(files):
                archive.add(path, arcname="copypaste/" + path.relative_to(ROOT).as_posix(), recursive=False)
        temporary.rename(output)
    finally:
        temporary.unlink(missing_ok=True)
    digest = hashlib.sha256()
    with output.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    output.with_name(output.name + ".sha256").write_text(f"{digest.hexdigest()}  {output.name}\n")
    print(f"Packaged {len(files)} files, {output.stat().st_size} bytes: {output}")


if __name__ == "__main__":
    main()
