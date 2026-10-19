#!/usr/bin/env python3
"""Moved into the e2e package; this forwards to
`python3 -m e2e kernel_set --author reference-scalar --rows-abi` (same arguments)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from e2e.cli import main  # noqa: E402

raise SystemExit(main(["kernel_set", "--author", "reference-scalar", "--rows-abi", *sys.argv[1:]]))
