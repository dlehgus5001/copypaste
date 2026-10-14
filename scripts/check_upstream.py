#!/usr/bin/env python3
"""Check that vendored official source files still match the pinned revision."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "third_party/SOURCE_SHA256.json").read_text())
failures = []
for name, digest in manifest.items():
    path = root / "third_party/YOLOX" / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        failures.append(name)
if failures:
    raise SystemExit("Upstream file mismatch: " + ", ".join(failures))
print(f"Verified {len(manifest)} upstream source files unchanged.")
