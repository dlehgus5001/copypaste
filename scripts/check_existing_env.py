#!/usr/bin/env python3
"""Inspect the active Python and plan local, missing-only dependency installation.

No packages are installed or replaced. A successful --plan writes a hashed lock
for manual pip --no-index --no-deps installation using the same interpreter.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from urllib.parse import unquote, urlsplit

try:
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    from packaging.version import Version
except ImportError:
    # pip already carries these parsers; inspecting need not install packaging.
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.utils import canonicalize_name
    from pip._vendor.packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]
CV_PROVIDERS = (
    "opencv-python", "opencv-python-headless",
    "opencv-contrib-python", "opencv-contrib-python-headless",
)


def read_requirements(path: Path) -> list[Requirement]:
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            requirement = Requirement(line)
            if requirement.url or requirement.extras:
                raise ValueError("Only plain package requirements are supported")
            if not requirement.marker or requirement.marker.evaluate():
                result.append(requirement)
    return result


def adapt_opencv(requirements, installed):
    """Reuse one existing cv2 provider instead of adding a second provider."""
    providers = [name for name in CV_PROVIDERS if name in installed]
    if len(providers) > 1:
        raise ValueError("Multiple cv2 providers installed: " + ", ".join(providers))
    result = []
    for requirement in requirements:
        if canonicalize_name(requirement.name) == "opencv-python-headless" and providers:
            requirement = Requirement(providers[0] + str(requirement.specifier))
        result.append(requirement)
    return result


def framework_probe(installed):
    result = {"gpu_validation": "not performed"}
    errors = []
    if "torch" not in installed:
        return result, errors
    try:
        import torch
        result.update(torch_version=torch.__version__, cuda_build=torch.version.cuda,
                      cuda_available=torch.cuda.is_available())
        if torch.version.cuda != "12.9":
            errors.append("Existing torch CUDA build must be 12.9 for this profile")
        if "torchvision" in installed:
            import torchvision
            result["torchvision_cuda_build"] = torchvision.version.cuda
            if torchvision.version.cuda != 12090:
                errors.append("Existing torchvision CUDA build must be 12.9 for this profile")
            boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.], [20., 20., 30., 30.]])
            kept = torchvision.ops.nms(boxes, torch.tensor([.9, .8, .7]), .5).tolist()
            if kept != [0, 2]:
                raise RuntimeError("Unexpected torchvision CPU NMS result")
            result.update(torchvision_version=torchvision.__version__, cpu_nms="passed")
    except Exception as error:
        errors.append(f"Existing framework import/operator failed: {type(error).__name__}: {error}")
    return result, errors


def runtime_probe(output):
    """Exercise binary interop and event writing without changing packages."""
    result, errors = {}, []
    try:
        import cv2
        import numpy as np
        import torch
        from pycocotools import mask as mask_api
        from torch.utils.tensorboard import SummaryWriter

        array = np.arange(6, dtype=np.float32).reshape(2, 3)
        np.testing.assert_array_equal(torch.from_numpy(array).numpy(), array)
        result["torch_numpy"] = "passed"
        resized = cv2.resize(array, (6, 4))
        if resized.shape != (4, 6):
            raise RuntimeError("Unexpected OpenCV resize result")
        mask = np.asfortranarray(np.array([[0, 1, 1], [1, 0, 1]], dtype=np.uint8))
        np.testing.assert_array_equal(mask_api.decode(mask_api.encode(mask)), mask)
        result["opencv_pycocotools"] = "passed"
        with tempfile.TemporaryDirectory(prefix="runtime-check-", dir=output) as temporary:
            with SummaryWriter(log_dir=temporary) as writer:
                writer.add_scalar("check/value", 1.0, 0)
                writer.flush()
            events = list(Path(temporary).glob("events.out.tfevents.*"))
            if not events or not all(path.stat().st_size for path in events):
                raise RuntimeError("TensorBoard did not write an event file")
        result["tensorboard_event_write"] = "passed"
        for name in ("loguru", "tqdm", "tabulate", "psutil", "thop"):
            importlib.import_module(name)
        result["auxiliary_imports"] = "passed"
    except Exception as error:
        errors.append(f"Existing runtime smoke failed: {type(error).__name__}: {error}")
    return result, errors


def missing_lock(plan, installed, manifest, wheelhouse):
    """Reject replacement plans and verify every proposed local wheel."""
    entries = plan.get("install", [])
    planned_names = [canonicalize_name(item["metadata"]["name"]) for item in entries]
    replacements = sorted(set(planned_names) & set(installed))
    if replacements:
        raise ValueError("Plan would replace installed packages: " + ", ".join(replacements))
    cuda_additions = sorted(name for name in planned_names if name.startswith("nvidia-") or name == "triton")
    if "torch" in installed and cuda_additions:
        raise ValueError("Existing Torch passed import, but pip metadata requests additional CUDA packages: "
                         + ", ".join(cuda_additions)
                         + ". Check Conda/system CUDA provisioning with the environment administrator; "
                         "a pip metadata gap does not establish missing runtime libraries.")
    known = {canonicalize_name(item["name"]): item for item in manifest["packages"]}
    lines = ["# Missing-only, verified local wheels. Use the same Python with --no-index --no-deps --require-hashes."]
    for item in entries:
        name = canonicalize_name(item["metadata"]["name"])
        expected = known.get(name)
        if expected is None or Version(item["metadata"]["version"]) != Version(expected["version"]):
            raise ValueError(f"Planned package is absent from the verified manifest: {name}")
        url = urlsplit(item["download_info"]["url"])
        path = Path(unquote(url.path)).resolve()
        if url.scheme != "file" or url.netloc or path.parent != wheelhouse.resolve() or path.name != expected["filename"]:
            raise ValueError(f"Planned wheel is not the expected local file: {name}")
        if not path.is_file() or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Missing wheel or size mismatch: {path.name}")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected["sha256"]:
            raise ValueError(f"Wheel SHA256 mismatch: {path.name}")
        lines.append(f"{expected['name']}=={expected['version']} --hash=sha256:{expected['sha256']}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="Resolve and verify a missing-only local-wheel plan; never install")
    parser.add_argument("--include-build", action="store_true", help="Also inspect optional C++ extension build prerequisites")
    parser.add_argument("--output", "--output-dir", type=Path, default=ROOT / "artifacts/existing-env", help="Fresh report directory")
    parser.add_argument("--wheelhouse", type=Path, default=ROOT / "wheelhouse")
    args = parser.parse_args()
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("Output directory is not empty; choose a new --output directory to preserve existing files")
    args.output.mkdir(parents=True, exist_ok=True)
    installed = {canonicalize_name(dist.metadata["Name"]): dist.version
                 for dist in metadata.distributions() if dist.metadata.get("Name")}
    report = {"schema_version": 1, "python": platform.python_version(), "python_executable": sys.executable,
              "packages": [], "missing": [], "conflicts": [], "plan": "not requested"}
    try:
        if sys.version_info[:2] != (3, 12):
            report["conflicts"].append("This checked profile requires Python 3.12")
        if platform.system() != "Linux" or platform.machine() != "x86_64":
            report["conflicts"].append("This checked profile requires Linux x86_64")
        requirements = read_requirements(ROOT / "requirements.txt")
        if args.include_build:
            requirements.extend(read_requirements(ROOT / "requirements-build.txt"))
        requirements = adapt_opencv(requirements, installed)
        for requirement in requirements:
            name = canonicalize_name(requirement.name)
            version = installed.get(name)
            status = "missing" if version is None else "satisfied" if requirement.specifier.contains(version) else "conflict"
            report["packages"].append({"requirement": str(requirement), "name": name,
                                       "version": version, "status": status})
            if status == "missing":
                report["missing"].append(name)
            elif status == "conflict":
                report["conflicts"].append(f"{name} {version} does not satisfy {requirement.specifier}")
        report["framework"], framework_errors = framework_probe(installed)
        report["conflicts"].extend(framework_errors)
        if not report["conflicts"] and not report["missing"]:
            report["runtime_checks"], runtime_errors = runtime_probe(args.output)
            report["conflicts"].extend(runtime_errors)
        if args.plan and not report["conflicts"]:
            if not report["missing"]:
                (args.output / "requirements-missing.txt").write_text(
                    "# No installation needed; the checked runtime profile is already installed.\n", encoding="utf-8")
                report.update(plan="no installation needed", planned_packages=[])
            else:
                report["plan"] = "blocked"
                if not args.wheelhouse.is_dir():
                    raise ValueError(f"Local wheelhouse directory is missing: {args.wheelhouse}; no network access will be attempted")
                resolved = args.output / "requirements-resolved.txt"
                resolved.write_text("\n".join(map(str, requirements)) + "\n", encoding="utf-8")
                plan_path = args.output / "pip-plan.json"
                command = [sys.executable, "-m", "pip", "--isolated", "--disable-pip-version-check",
                           "--cache-dir", str(args.output / "pip-cache"), "install", "--dry-run", "--no-input",
                           "--no-index", "--find-links", str(args.wheelhouse.resolve()), "--only-binary=:all:",
                           "--report", str(plan_path), "-r", str(resolved)]
                # --isolated still reads global/site pip configuration. Disable
                # those files too, so inherited remote find-links are excluded.
                completed = subprocess.run(command, capture_output=True, text=True,
                                           env={**os.environ, "PIP_CONFIG_FILE": os.devnull})
                (args.output / "pip-plan.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
                if completed.returncode:
                    raise ValueError("Offline pip resolution failed; a required wheel or compatible dependency is unavailable. "
                                     f"See {args.output / 'pip-plan.log'}. Installed packages were not changed.")
                plan = json.loads(plan_path.read_text(encoding="utf-8"))
                manifest = json.loads((ROOT / "wheelhouse-manifest.json").read_text(encoding="utf-8"))
                lock = missing_lock(plan, installed, manifest, args.wheelhouse)
                (args.output / "requirements-missing.txt").write_text(lock, encoding="utf-8")
                report.update(plan="safe missing-only plan", planned_packages=[item["metadata"]["name"] for item in plan["install"]])
        elif args.plan:
            report["plan"] = "blocked by existing conflicts"
    except (ValueError, OSError, KeyError) as error:
        report["conflicts"].append(str(error))
        if args.plan:
            report["plan"] = "blocked"
    report["status"] = "blocked" if report["conflicts"] else "missing dependencies" if report["missing"] else "checked profile satisfied"
    (args.output / "environment.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["conflicts"]:
        return 2
    return 0 if args.plan or not report["missing"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
