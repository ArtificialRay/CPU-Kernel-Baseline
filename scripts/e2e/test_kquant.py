#!/usr/bin/env python3
"""Moved to e2e/adapters/llamacpp/test_kquant.py; this runs it."""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
runpy.run_path(str(Path(__file__).resolve().parents[2] / "e2e" / "adapters" / "llamacpp" / "test_kquant.py"), run_name="__main__")
