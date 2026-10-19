#!/usr/bin/env python3
"""`generate` against the published warehouse: the definitions and solutions it writes for
qwen3.5-4b must be byte-identical to the ones already in bench-trace/ (they were produced by
this very generator). Workloads are not compared: capture/calibrate rewrote those afterwards
(real activations, SQNR tags).

Skipped when bench-trace/ doesn't hold the published qwen3.5-4b set.

Run:  .venv/bin/python -m unittest e2e.adapters.llamacpp.test_generate   (from the repo root)
"""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from e2e.adapters.llamacpp.generate import write_definitions

REPO = Path(__file__).resolve().parents[3]
INVENTORY = Path(__file__).resolve().parent / "testdata" / "qwen3.5-4b.inventory.json"
PUBLISHED = REPO / "bench-trace"


class GenerateGoldenTest(unittest.TestCase):
    def test_definitions_and_solutions_match_the_published_ones(self):
        if not (PUBLISHED / "definitions" / "gemm" / "gemm_ggml_q4_K_n1024_k2560.json").exists():
            self.skipTest("bench-trace/ doesn't hold the published qwen3.5-4b definitions")
        root = Path(tempfile.mkdtemp())
        with contextlib.redirect_stdout(io.StringIO()):
            written = write_definitions(INVENTORY, "qwen3.5-4b", root)
        self.assertGreater(written, 0)
        compared = 0
        for new in sorted(root.rglob("*.json")):
            old = PUBLISHED / new.relative_to(root)
            self.assertTrue(old.exists(), f"{old} is not published")
            self.assertEqual(new.read_bytes(), old.read_bytes(), f"{new.relative_to(root)} differs")
            compared += 1
        self.assertGreaterEqual(compared, 30)      # 11 definitions + 22 solutions


if __name__ == "__main__":
    unittest.main()
