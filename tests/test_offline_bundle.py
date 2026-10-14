"""Transport integrity and no-network lock format guards, using tiny fixtures."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts" / "verify_bundle.py"
    specification = importlib.util.spec_from_file_location("offline_bundle_verifier", path)
    verifier = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(verifier)
    payload = b"checksum fixture payload"
    sha256 = hashlib.sha256(payload).hexdigest()
    filename = "fixture-1.0-py3-none-any.whl"
    (tmp_path / "wheelhouse").mkdir()
    (tmp_path / "wheelhouse" / filename).write_bytes(payload)
    package = {"name": "fixture", "version": "1.0", "filename": filename,
               "sha256": sha256, "bytes": len(payload)}
    manifest = {"packages": [package], "total_bytes": len(payload)}
    (tmp_path / "wheelhouse-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "requirements-lock.txt").write_text(f"fixture==1.0 --hash=sha256:{sha256}\n", encoding="utf-8")
    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["verify_bundle.py", "--wheels-only"])
    return verifier, tmp_path, sha256


def test_valid_local_bundle(bundle, capsys):
    verifier, _, _ = bundle
    verifier.main()
    assert "Verified 1 wheels" in capsys.readouterr().out


@pytest.mark.parametrize("requirement", [
    "fixture @ https://example.invalid/fixture.whl --hash=sha256:{sha256}",
    "--index-url https://example.invalid",
    "fixture>=1.0 --hash=sha256:{sha256}",
    "fixture==1.0",
])
def test_nonlocal_or_unfrozen_lock_rejected(bundle, requirement):
    verifier, root, sha256 = bundle
    (root / "requirements-lock.txt").write_text(requirement.format(sha256=sha256) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="only exact named requirements"):
        verifier.main()


def test_changed_wheel_content_rejected(bundle):
    verifier, root, _ = bundle
    wheel = next((root / "wheelhouse").glob("*.whl"))
    # Preserve byte length so the SHA256 comparison detects the corruption.
    wheel.write_bytes(b"X" * wheel.stat().st_size)
    with pytest.raises(SystemExit, match="Wheel integrity failed"):
        verifier.main()


def test_missing_wheel_rejected(bundle):
    verifier, root, _ = bundle
    next((root / "wheelhouse").glob("*.whl")).unlink()
    with pytest.raises(SystemExit, match="Wheelhouse file set differs"):
        verifier.main()
