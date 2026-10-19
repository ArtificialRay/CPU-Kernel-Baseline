#!/usr/bin/env python3
"""measure() and render_report(), with fake llama-bench / llama-perplexity executables
(no model, no Graviton box).

Run:  .venv/bin/python -m unittest e2e.adapters.llamacpp.test_measure   (from the repo root)
"""
import contextlib
import io
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

from e2e.adapters.llamacpp.measure import measure
from e2e.adapters.llamacpp.report import render_report
from e2e.errors import StepFailed

TESTDATA = Path(__file__).resolve().parent / "testdata"

FAKE_BENCH = """#!/bin/sh
# prints one pp and one tg result like `llama-bench -o json`, and, when the override log is
# on, a "hit" line for every shape in $FAKE_HITS (type:K:N,...) like the real hook
if [ "$ARMBENCH_OVERRIDE_LOG" = "1" ]; then
  for h in $(echo "$FAKE_HITS" | tr ',' ' '); do
    echo "[armbench-override] hit mul_mat type=$(echo $h | cut -d: -f1) K=$(echo $h | cut -d: -f2) N=$(echo $h | cut -d: -f3) (M=1)" >&2
  done
fi
echo '[{"n_prompt": 16, "n_gen": 0, "avg_ts": %(pp)s}, {"n_prompt": 0, "n_gen": 4, "avg_ts": %(tg)s}]'
"""
FAKE_PPL = """#!/bin/sh
echo "perplexity: 1.0 seconds per pass" >&2
echo "Final estimate: PPL = %s +/- 0.5" >&2
"""


def executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


class MeasureTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.model = self.tmp / "model.gguf"
        self.model.write_bytes(b"GGUF")
        self.builds = {}
        for name, pp, tg, ppl in (("stock", 40.0, 10.0, "6.00"), ("agent", 20.0, 5.0, "6.06")):
            root = self.tmp / f"build-{name}"
            executable(root / "bin" / "llama-bench", FAKE_BENCH % {"pp": pp, "tg": tg})
            executable(root / "bin" / "llama-perplexity", FAKE_PPL % ppl)
            self.builds[name] = root
        self.manifest = self.tmp / "manifest.json"
        self.manifest.write_text(json.dumps({"kernels": [{"op": "mul_mat", "type": "q4_K", "K": 2560, "N": 1024}]}))
        self.text = self.tmp / "wiki.raw"
        self.text.write_text("text")
        self.saved = os.environ.get("FAKE_HITS")
        os.environ["FAKE_HITS"] = "q4_K:2560:1024"
        self.addCleanup(lambda: os.environ.pop("FAKE_HITS") if self.saved is None else os.environ.update(FAKE_HITS=self.saved))

    def run_measure(self, **kw):
        out = self.tmp / "results.json"
        with contextlib.redirect_stdout(io.StringIO()):
            return measure(self.model, self.builds, overrides={"agent": self.manifest}, fa="off", out=out,
                           pp=16, tg=4, reps=2, **kw), out

    def test_results_summary_hook_and_perplexity(self):
        results, out = self.run_measure(perplexity=self.text, ppl_chunks=1)
        self.assertEqual(json.loads(out.read_text())["summary"], results["summary"])
        self.assertEqual(results["summary"]["stock/t4/pp"]["median"], 40.0)
        self.assertEqual(results["summary"]["agent/t4/tg"]["speedup_vs_stock_median"], 0.5)
        self.assertEqual(results["hook"], {"agent": {"expected": 1, "hit": 1}})
        self.assertEqual(len(results["samples"]), 2 * 2 * 2)             # reps x builds x (pp, tg)
        self.assertIn("PPL = 6.06", results["perplexity"]["agent"])
        table = render_report(results)
        self.assertIn("agent                 0.50x   0.50x", table)
        self.assertIn("agent                 6.060  (+1.0% vs stock)", table)

    def test_no_perplexity_without_a_text_file(self):
        results, _ = self.run_measure()
        self.assertEqual(results["perplexity"], {})

    def test_an_override_that_is_not_hit_stops_the_run(self):
        os.environ["FAKE_HITS"] = "q4_K:9999:9999"
        with self.assertRaises(StepFailed) as ctx:
            self.run_measure()
        self.assertIn("override hook check failed", str(ctx.exception))

    def test_unknown_build_names_are_rejected(self):
        with self.assertRaises(StepFailed):
            with contextlib.redirect_stdout(io.StringIO()):
                measure(self.model, self.builds, overrides={"nope": self.manifest}, fa="off", out=self.tmp / "r.json")
        with self.assertRaises(StepFailed):
            with contextlib.redirect_stdout(io.StringIO()):
                measure(self.model, {"stock": self.tmp / "missing"}, fa="off", out=self.tmp / "r.json")


class ReportTest(unittest.TestCase):
    def test_table_matches_what_the_original_results_table_script_printed(self):
        """example.table.txt is the original scripts/e2e/results_table.py's output for this
        results file (a real run on a c8g.xlarge), checked byte for byte when it was made."""
        expected = (TESTDATA / "e2e_results.example.table.txt").read_text()
        self.assertEqual(render_report(TESTDATA / "e2e_results.example.json") + "\n", expected)


if __name__ == "__main__":
    unittest.main()
