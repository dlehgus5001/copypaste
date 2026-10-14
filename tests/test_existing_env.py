"""Ensure existing-environment planning cannot silently replace packages."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SPEC = importlib.util.spec_from_file_location(
    "check_existing_env", Path(__file__).resolve().parents[1] / "scripts/check_existing_env.py")
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def test_existing_gui_opencv_is_reused():
    requirements = [checker.Requirement("numpy>=1.26.4,<3"),
                    checker.Requirement("opencv-python-headless>=4.11,<5")]
    adapted = checker.adapt_opencv(requirements, {"opencv-python": "4.13.0"})
    assert [req.name for req in adapted] == ["numpy", "opencv-python"]
    assert adapted[1].specifier.contains("4.13.0")
    assert adapted[1].specifier.contains("4.11.0.86")
    assert not adapted[1].specifier.contains("4.8.0")


def test_overlapping_opencv_providers_are_blocked():
    with pytest.raises(ValueError, match="Multiple cv2 providers"):
        checker.adapt_opencv([], {"opencv-python": "4.13.0", "opencv-python-headless": "4.14.0"})


def test_runtime_cuda_local_suffix_matches_torch_base_requirement():
    assert checker.Requirement("torch==2.8.0").specifier.contains("2.8.0+cu129")


def test_cpu_only_torchvision_is_not_mistaken_for_cuda_profile(monkeypatch):
    import torchvision

    monkeypatch.setattr(torchvision.version, "cuda", None)
    report, errors = checker.framework_probe({"torch": "2.8.0+cu129", "torchvision": "0.23.0+cpu"})
    assert report["cpu_nms"] == "passed"
    assert "Existing torchvision CUDA build must be 12.9 for this profile" in errors


def test_replacement_of_transitive_installed_package_is_blocked(tmp_path):
    plan = {"install": [{"metadata": {"name": "protobuf", "version": "7.36.2"}}]}
    with pytest.raises(ValueError, match="replace installed packages: protobuf"):
        checker.missing_lock(plan, {"protobuf": "4.25.0"}, {"packages": []}, tmp_path)


def test_existing_torch_cuda_metadata_gap_is_distinguished(tmp_path):
    plan = {"install": [{"metadata": {"name": "nvidia-cuda-runtime-cu12", "version": "12.9.79"}}]}
    with pytest.raises(ValueError, match="metadata gap does not establish missing runtime"):
        checker.missing_lock(plan, {"torch": "2.8.0+cu129"}, {"packages": []}, tmp_path)


def wheel_fixture(tmp_path):
    wheel = tmp_path / "loguru-0.7.3-py3-none-any.whl"
    wheel.write_bytes(b"wheel content for checksum guard")
    package = {"name": "loguru", "version": "0.7.3", "filename": wheel.name,
               "bytes": wheel.stat().st_size, "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest()}
    plan = {"install": [{"metadata": {"name": "loguru", "version": "0.7.3"},
                         "download_info": {"url": wheel.as_uri()}}]}
    return wheel, package, plan


def test_missing_only_lock_contains_verified_candidate(tmp_path):
    _, package, plan = wheel_fixture(tmp_path)
    lock = checker.missing_lock(plan, {"torch": "2.8.0+cu129"}, {"packages": [package]}, tmp_path)
    assert f"loguru==0.7.3 --hash=sha256:{package['sha256']}\n" in lock
    assert "torch==" not in lock


def test_corrupted_candidate_is_rejected(tmp_path):
    wheel, package, plan = wheel_fixture(tmp_path)
    wheel.write_bytes(b"X" * package["bytes"])
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        checker.missing_lock(plan, {}, {"packages": [package]}, tmp_path)


def test_online_candidate_is_rejected(tmp_path):
    _, package, plan = wheel_fixture(tmp_path)
    plan["install"][0]["download_info"]["url"] = "https://example.invalid/" + package["filename"]
    with pytest.raises(ValueError, match="not the expected local file"):
        checker.missing_lock(plan, {}, {"packages": [package]}, tmp_path)


def test_satisfied_environment_produces_empty_lock(tmp_path):
    lock = checker.missing_lock({"install": []}, {"torch": "2.8.0+cu129"}, {"packages": []}, tmp_path)
    assert all(line.startswith("#") for line in lock.splitlines())


def test_actual_preinstalled_runtime_needs_no_wheelhouse(tmp_path):
    output = tmp_path / "report"
    absent = tmp_path / "absent-wheelhouse"
    completed = subprocess.run(
        [sys.executable, str(Path(checker.__file__)), "--plan", "--wheelhouse", str(absent),
         "--output", str(output)], capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads((output / "environment.json").read_text())
    assert report["plan"] == "no installation needed"
    assert report["planned_packages"] == []
    assert report["runtime_checks"]["torch_numpy"] == "passed"
    assert report["runtime_checks"]["opencv_pycocotools"] == "passed"
    assert report["runtime_checks"]["tensorboard_event_write"] == "passed"
    assert not absent.exists()
    assert not (output / "pip-plan.json").exists()
    assert not (output / "pip-cache").exists()
