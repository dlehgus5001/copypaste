#!/usr/bin/env python3
"""Record nonsecret host, compiler, Python, CUDA and PyTorch diagnostics."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def command(name, arguments):
    local = Path(sys.executable).parent / name
    path = str(local) if local.is_file() and os.access(local, os.X_OK) else shutil.which(name)
    if not path:
        return {"available": False}
    result = subprocess.run([path, *arguments], capture_output=True, text=True, timeout=30)
    return {"available": True, "path": path, "returncode": result.returncode,
            "output": (result.stdout + result.stderr).strip()[:6000]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = {"python": sys.version, "python_executable": sys.executable,
            "platform": platform.platform(), "machine": platform.machine(),
            "cpu_count": os.cpu_count(), "disk": dict(zip(("total", "used", "free"), shutil.disk_usage(Path.cwd()))),
            "os_release": Path("/etc/os-release").read_text(encoding="utf-8") if Path("/etc/os-release").is_file() else None,
            "installed_packages": sorted(
                ({"name": package.metadata["Name"], "version": package.version} for package in importlib.metadata.distributions()),
                key=lambda package: package["name"].lower()),
            "g++": command("g++", ["--version"]), "gcc": command("gcc", ["--version"]),
            "nvcc": command("nvcc", ["--version"]),
            "nvidia-smi": command("nvidia-smi", ["--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
            "cmake": command("cmake", ["--version"]), "ninja": command("ninja", ["--version"])}
    try:
        import torch
        data["torch"] = {"version": torch.__version__, "cuda_runtime": torch.version.cuda,
                         "cuda_available": torch.cuda.is_available(), "cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
                         "device_count": torch.cuda.device_count(), "cudnn_version": torch.backends.cudnn.version(),
                         "compiled_cuda_arch_flags": torch._C._cuda_getArchFlags(),
                         "devices": [{"name": torch.cuda.get_device_name(index),
                                      "capability": list(torch.cuda.get_device_capability(index)),
                                      "memory_bytes": torch.cuda.get_device_properties(index).total_memory}
                                     for index in range(torch.cuda.device_count())]}
        data["torchvision"] = importlib.metadata.version("torchvision")
        try:
            import yolox
            data["yolox"] = {"version": yolox.__version__, "module_path": str(Path(yolox.__file__).resolve())}
        except ImportError:
            data["yolox"] = {"available": False}
    except ImportError:
        data["torch"] = {"available": False}
    rendered = json.dumps(data, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
