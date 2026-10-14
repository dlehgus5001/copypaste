"""Locate the repository and its vendored implementation without package downloads."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "third_party" / "YOLOX"


def activate_vendor():
    if not (VENDOR / "yolox" / "__init__.py").is_file():
        raise RuntimeError(f"Pinned YOLOX source is missing: {VENDOR}")
    vendor = str(VENDOR)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    return VENDOR


def local_path(value, *, must_exist=False):
    """Configuration paths are local paths, relative to the checkout root."""
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError("A nonempty local filesystem path is required")
    if "://" in str(value):
        raise ValueError("Only local filesystem paths are supported; downloads are disabled")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


activate_vendor()
