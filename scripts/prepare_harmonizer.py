#!/usr/bin/env python3
"""ONLINE preparation only: fetch pinned Harmonizer source/weights for offline use."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def prepare(output):
    output = Path(output).expanduser()
    if not output.is_absolute():
        output = ROOT / output
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing folder: {output}")
    manifest = json.loads((ROOT / "configs/harmonizer-source.json").read_text())
    base = "https://raw.githubusercontent.com/ZHKKKe/Harmonizer/" + manifest["revision"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="harmonizer-download-", dir=output.parent) as temporary:
        staging = Path(temporary) / "Harmonizer"
        for relative, expected in manifest["files"].items():
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {relative}", flush=True)
            digest = hashlib.sha256()
            with urllib.request.urlopen(base + "/" + relative, timeout=60) as response, target.open("xb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f"SHA256 mismatch: {relative}")
        (staging / "SOURCE.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(output)
    print(f"Prepared pinned source and checkpoint: {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="models/Harmonizer")
    prepare(parser.parse_args().output)
