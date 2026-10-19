#!/usr/bin/env python3
"""The framework-neutral pieces in e2e.common, and the default `calibrate` step of the adapter base class.

Run:  .venv/bin/python -m unittest e2e.common.test_common   (from the repo root)
"""
import json
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from e2e.adapters.base import FrameworkAdapter, Step
from e2e.common import calibration
from e2e.common.build import compile_shared, undefined_symbols
from e2e.common.guards import check_guards
from e2e.common.solutions import load_author, source_text
from e2e.context import E2EContext, Options
from e2e.errors import StepFailed


def solution(kernel: str) -> dict:
    return {"sources": [{"path": "gemm.cpp", "content": "// harness\n"}, {"path": "kernel.cpp", "content": kernel}]}


class SolutionsTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())

    def put(self, name, sol, author="a", op_type="gemm"):
        d = self.root / "ds" / author / op_type
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.json").write_text(json.dumps(sol))

    def test_load_author_skips_session_intermediates(self):
        self.put("k", solution("// best\n"))
        self.put("k_current", solution("// last compile\n"))
        self.put("k_best", solution("// mid-rename\n"))
        got = load_author("ds", "a", self.root)
        self.assertEqual(list(got), ["k"])
        self.assertEqual(source_text(got["k"]), "// best\n")
        self.assertEqual(source_text(got["k"], "gemm.cpp"), "// harness\n")

    def test_load_author_reads_the_requested_op_type_and_fails_when_missing(self):
        self.put("k", solution("// x\n"), op_type="moe")
        self.assertEqual(list(load_author("ds", "a", self.root, op_type="moe")), ["k"])
        with self.assertRaises(StepFailed):
            load_author("ds", "a", self.root)                    # no gemm/ directory
        with self.assertRaises(StepFailed):
            load_author("ds", "nobody", self.root)


class GuardsTest(unittest.TestCase):
    sols = {"a": solution("// new\n"), "b": solution("// same\n")}
    ref = {"a": solution("// new\n"), "b": solution("// same\n")}

    def run_guards(self, **kw):
        return check_guards(self.sols, ["a", "b"], self.ref, digest=lambda t: t.strip(), **kw)

    def test_everything_holds(self):
        self.assertEqual(self.run_guards(expect_same=["a", "b"], expect_count=2, forbid=["nope"]), [])

    def test_a_stale_set_is_reported(self):
        fail = self.run_guards(expect_changed=["a"])
        self.assertEqual(len(fail), 1)
        self.assertIn("BYTE-IDENTICAL", fail[0])

    def test_count_forbid_and_missing_names(self):
        fail = self.run_guards(expect_count=3, forbid=["same"], expect_same=["zzz"], expect_changed=["zzz"])
        self.assertTrue(any("expected 3" in f for f in fail))
        self.assertTrue(any("forbidden pattern" in f for f in fail))
        self.assertEqual(sum("absent" in f for f in fail), 2)

    def test_digest_decides_what_counts_as_equal(self):
        self.assertEqual(check_guards(self.sols, ["a"], {"a": solution("// NEW\n")}, digest=str.lower,
                                      expect_same=["a"]), [])


class BuildTest(unittest.TestCase):
    def test_compile_shared_builds_the_command_the_harness_used(self):
        tmp = Path(tempfile.mkdtemp())
        seen = tmp / "args"
        cxx = tmp / "fake-cxx"
        cxx.write_text(f'#!/bin/sh\necho "$@" > {seen}\n')
        cxx.chmod(cxx.stat().st_mode | stat.S_IEXEC)
        compile_shared(str(cxx), "-march=armv9-a", tmp, [tmp / "k.cpp", tmp / "g.cpp"], tmp / "out.so")
        self.assertEqual(seen.read_text().split(),
                         ["-shared", "-fPIC", "-O3", "-march=armv9-a", "-std=c++14", "-I", str(tmp),
                          str(tmp / "k.cpp"), str(tmp / "g.cpp"), "-o", str(tmp / "out.so")])

    @unittest.skipUnless(shutil.which("gcc") and shutil.which("nm"), "needs gcc and nm")
    def test_undefined_symbols_filters_by_prefix(self):
        tmp = Path(tempfile.mkdtemp())
        (tmp / "k.c").write_text("extern int ggml_a(void); extern int other_b(void);"
                                 "int f(void){return ggml_a()+other_b();}\n")
        subprocess.run(["gcc", "-shared", "-fPIC", str(tmp / "k.c"), "-o", str(tmp / "k.so")], check=True)
        self.assertEqual(undefined_symbols(tmp / "k.so", "ggml_"), ["ggml_a"])
        self.assertEqual(undefined_symbols(tmp / "k.so", "other_"), ["other_b"])


class DefaultCalibrateTest(unittest.TestCase):
    """The base class's `calibrate` is the same for every framework; the adapter supplies two names."""

    class Fw(FrameworkAdapter):
        name = "fw"
        reference_author = "my-reference"
        baseline_author = "my-baseline"
        prepare_steps = (Step("calibrate", box=True),)
        evaluate_steps = ()

        def fetch_header(self, model_config, dest):
            return dest

    def test_uses_the_adapters_authors_and_the_models_tag(self):
        ctx = E2EContext.create("fw", "qwen3.5-4b", "sve2", Path(tempfile.mkdtemp()))
        with mock.patch.object(calibration, "calibrate_sqnr_floors", return_value=(6, 0)) as fn:
            self.Fw().calibrate(ctx, Options(root=Path("/warehouse")))
        fn.assert_called_once_with(Path("/warehouse"), author="my-reference", baseline_author="my-baseline",
                                   tag="e2e:qwen3.5-4b")

    def test_problems_fail_the_step(self):
        ctx = E2EContext.create("fw", "qwen3.5-4b", "sve2", Path(tempfile.mkdtemp()))
        with mock.patch.object(calibration, "calibrate_sqnr_floors", return_value=(4, 2)):
            with self.assertRaises(StepFailed):
                self.Fw().calibrate(ctx, Options())


if __name__ == "__main__":
    unittest.main()
