#!/usr/bin/env python3
"""Resolve official wheels once and save a fully hashed offline dependency closure.

Run on the online Linux x86_64 Python 3.12 staging machine. The generated lock
contains package names, versions and hashes, never direct URLs. Provenance is
retained separately in wheelhouse-manifest.json and artifacts/resolution.json.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit, urlunsplit
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_HOSTS = {"files.pythonhosted.org", "pypi.org", "download.pytorch.org", "download-r2.pytorch.org"}


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.urls.extend(value for key, value in attrs if key == "href" and value)


def official_framework_wheels() -> list[str]:
    urls = []
    for name, version in (("torch", "2.8.0+cu129"), ("torchvision", "0.23.0+cu129")):
        with urlopen(f"https://download.pytorch.org/whl/cu129/{name}/", timeout=60) as response:
            parser = Links()
            parser.feed(response.read().decode("utf-8"))
        target = f"{name}-{version}-cp312-cp312-manylinux_2_28_x86_64.whl"
        matches = [url for url in parser.urls if unquote(Path(urlsplit(url).path).name) == target]
        if len(matches) != 1 or not urlsplit(matches[0]).fragment.startswith("sha256="):
            raise SystemExit(f"No unique checksum-bearing official wheel found: {target}")
        urls.extend(matches)
    return urls


def run(command: list[str], log: Path) -> None:
    with log.open("w", encoding="utf-8") as output:
        completed = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
    if completed.returncode:
        print(log.read_text(encoding="utf-8")[-10000:], file=sys.stderr)
        raise SystemExit(f"Command failed ({completed.returncode}); see {log}")


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-resolution", action="store_true", help="Reuse the saved resolution or wheelhouse manifest rather than resolve newer packages")
    parser.add_argument("--jobs", type=int, default=4, help="Independent download streams (1 to 8, default 4)")
    args = parser.parse_args()
    if not 1 <= args.jobs <= 8:
        raise SystemExit("--jobs must be between 1 and 8")
    if sys.version_info[:2] != (3, 12) or platform.system() != "Linux" or platform.machine() != "x86_64":
        raise SystemExit("Collection requires Linux x86_64 and Python 3.12. Collect a separate lock and wheelhouse for other targets.")
    artifacts = ROOT / "artifacts"
    wheels = ROOT / "wheelhouse"
    artifacts.mkdir(exist_ok=True)
    wheels.mkdir(exist_ok=True)
    report = artifacts / "resolution.json"
    pip = [sys.executable, "-m", "pip", "--isolated", "--cache-dir", str(ROOT / ".pip-cache")]
    if not args.reuse_resolution:
        print("Resolving Python 3.12 wheels from PyPI and the official PyTorch cu129 index", flush=True)
        # Only framework wheels come from PyTorch. Resolve every other package
        # through PyPI, whose NVIDIA wheels match torch's exact dependencies.
        # The PyTorch index can advertise a separate NVIDIA index; do not mix it.
        framework_wheels = official_framework_wheels()
        run(pip + ["install", "--dry-run", "--ignore-installed", "--only-binary=:all:", "--report", str(report),
            "--index-url", "https://pypi.org/simple",
            "-r", str(ROOT / "requirements-core.txt"), *framework_wheels], artifacts / "wheel-resolution.log")
    if args.reuse_resolution and not report.is_file():
        manifest_path = ROOT / "wheelhouse-manifest.json"
        if not manifest_path.is_file():
            raise SystemExit("No saved resolution or wheelhouse manifest to reuse")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        resolved = {"install": [{"metadata": {"name": p["name"], "version": p["version"]},
                                 "download_info": {"url": p["source_url"], "archive_info": {"hashes": {"sha256": p["sha256"]}}}}
                                for p in previous["packages"]]}
    else:
        resolved = json.loads(report.read_text(encoding="utf-8"))
    packages = []
    for item in resolved["install"]:
        info = item["download_info"]
        url = info["url"]
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS or parsed.username or parsed.password:
            raise SystemExit(f"Unapproved or insecure package source for {item['metadata']['name']}")
        filename = unquote(Path(parsed.path).name)
        if not filename.endswith(".whl"):
            raise SystemExit(f"Source distributions are not allowed: {filename}")
        sha256 = info["archive_info"]["hashes"]["sha256"]
        packages.append({"name": item["metadata"]["name"], "version": item["metadata"]["version"],
                         "filename": filename, "source_url": url, "sha256": sha256})
    packages.sort(key=lambda item: item["name"].lower().replace("_", "-"))
    (ROOT / "requirements-lock.txt").write_text(
        "# Linux x86_64 / CPython 3.12. Generated from official HTTPS package sources.\n"
        + "\n".join(f"{p['name']}=={p['version']} --hash=sha256:{p['sha256']}" for p in packages) + "\n", encoding="utf-8")
    download_lock = artifacts / "requirements-download.txt"
    download_lock.write_text("\n".join(f"{p['source_url']} --hash=sha256:{p['sha256']}" for p in packages) + "\n", encoding="utf-8")
    print(f"Downloading {len(packages)} checksum-verified wheels", flush=True)
    download_logs = artifacts / "wheel-downloads"
    download_logs.mkdir(exist_ok=True)

    def download(package):
        destination = wheels / package["filename"]
        if destination.is_file() and digest(destination) == package["sha256"]:
            return package["filename"]
        split = urlsplit(package["source_url"])
        verified_url = urlunsplit((split.scheme, split.netloc, split.path, split.query, "sha256=" + package["sha256"]))
        run(pip + ["download", "--no-index", "--no-deps", "--require-hashes", "--only-binary=:all:",
                   "--dest", str(wheels), verified_url], download_logs / (package["name"] + ".log"))
        return package["filename"]

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(download, package) for package in packages]
        for completed in as_completed(futures):
            print("Downloaded " + completed.result(), flush=True)
    filenames = {p["filename"] for p in packages}
    extra = {p.name for p in wheels.glob("*.whl")} - filenames
    if extra:
        raise SystemExit(f"Unexpected stale wheels: {sorted(extra)}; move them out before packaging")
    for package in packages:
        path = wheels / package["filename"]
        if digest(path) != package["sha256"]:
            raise SystemExit(f"Checksum mismatch: {path.name}")
        package["bytes"] = path.stat().st_size
    glibc_tags = [tuple(map(int, tag)) for package in packages
                  for tag in re.findall(r"manylinux_(\d+)_(\d+)", package["filename"])]
    minimum_glibc = ".".join(map(str, max(glibc_tags, default=(2, 17))))
    manifest = {"schema_version": 1, "target": {"python": "3.12", "os": "Linux", "architecture": "x86_64",
                "minimum_glibc": minimum_glibc,
                "torch": "2.8.0+cu129", "torchvision": "0.23.0+cu129", "cuda_runtime": "12.9"},
                "total_bytes": sum(p["bytes"] for p in packages), "packages": packages}
    (ROOT / "wheelhouse-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(packages)} wheels, {manifest['total_bytes'] / 1024**3:.2f} GiB. Run scripts/install_offline.sh.")


if __name__ == "__main__":
    main()
