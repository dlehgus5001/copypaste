#!/usr/bin/env python3
"""Standalone entry point; all inputs and outputs are local."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from offline_yolox.sample_data import main

if __name__ == "__main__":
    main()
