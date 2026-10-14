#!/usr/bin/env python3
"""Verify local wheel checksums, lock provenance and installed target versions."""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheels-only", action="store_true")
    parser.add_argument("--installed", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "wheelhouse-manifest.json").read_text(encoding="utf-8"))
    lock = (ROOT / "requirements-lock.txt").read_text(encoding="utf-8")
    rows = [line for line in lock.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if len(rows) != len(manifest["packages"]):
        raise SystemExit("Lock and manifest package counts disagree")
    for row in rows:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+==[^\s;]+ --hash=sha256:[a-f0-9]{64}", row):
            raise SystemExit("Offline lock must contain only exact named requirements and SHA256 hashes")
    actual = {p.name for p in (ROOT / "wheelhouse").glob("*.whl")}
    expected = {p["filename"] for p in manifest["packages"]}
    if actual != expected:
        raise SystemExit(f"Wheelhouse file set differs: {sorted(actual ^ expected)}")
    for package in manifest["packages"]:
        expected_row = f"{package['name']}=={package['version']} --hash=sha256:{package['sha256']}"
        if expected_row not in rows:
            raise SystemExit(f"Lock differs from manifest: {package['name']}")
        path = ROOT / "wheelhouse" / package["filename"]
        result = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                result.update(chunk)
        if path.stat().st_size != package["bytes"] or result.hexdigest() != package["sha256"]:
            raise SystemExit(f"Wheel integrity failed: {path.name}")
        if args.installed and metadata.version(package["name"]) != package["version"]:
            raise SystemExit(f"Installed version differs: {package['name']}")
    if args.installed:
        import torch
        import torchvision
        from torchvision.ops import nms
        import cv2
        import numpy as np
        import pycocotools.mask
        import tensorboard
        import yolox
        assert torch.__version__ == "2.8.0+cu129", torch.__version__
        assert torchvision.__version__ == "0.23.0+cu129", torchvision.__version__
        assert torch.version.cuda == "12.9", torch.version.cuda
        boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]])
        assert nms(boxes, torch.tensor([0.9, 0.8]), 0.5).tolist() == [0]
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        assert cv2.resize(image, (4, 4)).shape == (4, 4, 3)
        print(f"CPU imports/torchvision NMS passed; CUDA available={torch.cuda.is_available()}")
    print(f"Verified {len(expected)} wheels ({manifest['total_bytes'] / 1024**3:.2f} GiB) and hashed offline lock")


if __name__ == "__main__":
    main()
