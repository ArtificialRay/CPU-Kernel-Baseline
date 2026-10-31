#!/usr/bin/env python3
"""Benchmark.run_author: each solution of an author runs against ITS dataset's baseline author.

No kernels are built or run: bench.benchmark.run_solution_on_workloads is replaced by a recorder.

Run:  .venv/bin/python -m unittest bench.test_benchmark   (from the repo root)
"""
import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest import mock

from bench import benchmark, cli
from contracts import BASELINE_AUTHORS, baseline_author_for


def solution(name, definition, dataset, author):
    return SimpleNamespace(name=name, definition=definition, author=author,
                           dataset=SimpleNamespace(value=dataset))


def trace(ok=True, log=""):
    return SimpleNamespace(is_successful=lambda: ok, workload=SimpleNamespace(uuid="0123456789"),
                           evaluation=SimpleNamespace(status=SimpleNamespace(value="PASSED" if ok else "RUNTIME_ERROR"), log=log))


def no_builders():
    """Benchmark() asks for the builder registry, which needs a compiler this machine may not have."""
    return mock.patch.object(benchmark.BuilderRegistry, "get_instance",
                             return_value=SimpleNamespace(cleanup=lambda: None))


class FakeTraceSet:
    def __init__(self):
        sols = [
            solution("rs_conv", "conv_a", "ncnn", "reference-scalar"),
            solution("rs_loop", "loop_001", "simd-loop", "reference-scalar"),
            solution("rs_gemm", "gemm_a", "llama.cpp", "reference-scalar"),
            solution("rs_gone", "missing", "kleidiai", "reference-scalar"),
            solution("rs_nowl", "empty", "ncnn", "reference-scalar"),
            solution(BASELINE_AUTHORS["ncnn"] + "_conv", "conv_a", "ncnn", BASELINE_AUTHORS["ncnn"]),
        ]
        self.solutions = {}
        for s in sols:
            self.solutions.setdefault(s.definition, []).append(s)
        self.defs = {"conv_a": "conv2d", "loop_001": "loop_001", "gemm_a": "gemm", "empty": "conv2d"}
        self.traces = []

    def get_definition(self, name):
        return SimpleNamespace(name=name, op_type=self.defs[name]) if name in self.defs else None

    def get_workloads(self, name):
        return [] if name == "empty" else [SimpleNamespace(axes={"M": 1}), SimpleNamespace(axes={"M": 8})]

    def add_traces(self, traces):
        self.traces.extend(traces)


class RunAuthorTest(unittest.TestCase):
    def setUp(self):
        self.ts = FakeTraceSet()
        self.calls = []

        def runner(definition, solution, workloads, *, is_baseline, cfg, trace_set):
            self.calls.append((solution.name, is_baseline, cfg.baseline_author, len(workloads)))
            if solution.name == "rs_gemm":
                return [trace(), trace(ok=False, log="mismatch")]
            if solution.name == "rs_loop":
                raise RuntimeError("no builder")
            return [trace(), trace()]

        for patch in (mock.patch.object(benchmark, "run_solution_on_workloads", runner), no_builders()):
            patch.start()
            self.addCleanup(patch.stop)
        self.bench = benchmark.Benchmark(self.ts, benchmark.BenchmarkConfig())

    def test_each_solution_gets_the_baseline_author_of_its_dataset(self):
        results = self.bench.run_author("reference-scalar", isa="sve")
        by_name = {c[0]: c for c in self.calls}
        self.assertEqual(by_name["rs_conv"][2], BASELINE_AUTHORS["ncnn"])
        self.assertEqual(by_name["rs_gemm"][2], BASELINE_AUTHORS["llama.cpp"])
        self.assertEqual(by_name["rs_loop"][2], baseline_author_for("simd-loop", "sve"))
        # reference-scalar is a candidate in every dataset: none of them takes the baseline path
        self.assertFalse(any(c[1] for c in self.calls))
        self.assertEqual(len(results), 5)

    def test_the_baseline_author_itself_takes_the_baseline_path(self):
        self.bench.run_author(BASELINE_AUTHORS["ncnn"])
        self.assertEqual(self.calls, [(BASELINE_AUTHORS["ncnn"] + "_conv", True, BASELINE_AUTHORS["ncnn"], 2)])

    def test_failures_and_skips_are_reported_not_raised(self):
        by_name = {r.solution.name: r for r in self.bench.run_author("reference-scalar")}
        self.assertIn("no builder", by_name["rs_loop"].error)
        self.assertEqual(by_name["rs_gone"].skipped, "definition 'missing' not found")
        self.assertEqual(by_name["rs_nowl"].skipped, "no workloads")
        self.assertEqual([t.is_successful() for t in by_name["rs_gemm"].traces], [True, False])

    def test_filters_and_workload_axes(self):
        self.assertEqual([r.solution.name for r in self.bench.run_author("reference-scalar", dataset="ncnn")],
                         ["rs_conv", "rs_nowl"])
        # a solution whose definition is missing cannot be matched to an op_type, so it is still reported as skipped
        self.assertEqual([r.solution.name for r in self.bench.run_author("reference-scalar", op_type="gemm")],
                         ["rs_gone", "rs_gemm"])
        self.bench.run_author("reference-scalar", definition="conv_a", workload_filter={"M": 8})
        self.assertEqual(self.calls[-1][3], 1)

    def test_traces_are_only_persisted_on_request(self):
        self.bench.run_author("reference-scalar", dataset="ncnn")
        self.assertEqual(self.ts.traces, [])
        self.bench.run_author("reference-scalar", dataset="ncnn", dump_traces=True)
        self.assertEqual(len(self.ts.traces), 2)

    def test_run_solution_still_uses_the_configured_author_unless_told_otherwise(self):
        d, s = self.ts.get_definition("conv_a"), self.ts.solutions["conv_a"][1]      # the ncnn baseline solution
        self.bench.run_solution(d, s, self.ts.get_workloads("conv_a"), dump_traces=False)
        self.assertEqual(self.calls[-1][1:3], (False, benchmark.BenchmarkConfig().baseline_author) if s.author != benchmark.BenchmarkConfig().baseline_author else (True, s.author))
        self.bench.run_solution(d, s, self.ts.get_workloads("conv_a"), dump_traces=False, baseline_author=s.author)
        self.assertEqual(self.calls[-1][1:3], (True, s.author))


class BenchAuthorCommandTest(unittest.TestCase):
    def run_command(self, **kw):
        args = dict(root="r", author="reference-scalar", dataset=None, op_type=None, definition=None,
                    workload_axes=None, isa=None, dump_traces=False)
        args.update(kw)
        buf = io.StringIO()
        with mock.patch.object(cli.TraceSet, "from_path", return_value=FakeTraceSet()), no_builders(), \
                mock.patch.object(benchmark, "run_solution_on_workloads",
                                  lambda d, s, w, **k: [trace(), trace(ok=s.name != "rs_gemm", log="mismatch")]), \
                contextlib.redirect_stdout(buf):
            rc = cli.cmd_bench_author(SimpleNamespace(**args))
        return rc, buf.getvalue()

    def test_a_failing_workload_fails_the_command_and_names_the_solution(self):
        rc, text = self.run_command(dataset="llama.cpp")
        self.assertEqual(rc, 1)
        self.assertIn("[FAIL]", text)
        self.assertIn("1/2", text)
        self.assertIn("FAILED: rs_gemm", text)
        self.assertIn("mismatch", text)

    def test_all_passing_returns_zero_and_counts_workloads(self):
        rc, text = self.run_command(dataset="ncnn", definition="conv_a")
        self.assertEqual(rc, 0)
        self.assertIn("1/1 solutions fully passed; 2/2 workloads PASSED.", text)

    def test_no_author_and_no_dataset_exits(self):
        with self.assertRaises(SystemExit):
            self.run_command(author=None)

    def test_the_dataset_alone_picks_its_reference_author(self):
        rc, text = self.run_command(author=None, dataset="ncnn", definition="conv_a")
        self.assertEqual(rc, 0)

    def test_an_unknown_author_is_an_error_exit(self):
        rc, text = self.run_command(author="nobody")
        self.assertEqual(rc, 1)
        self.assertIn("No solutions of author 'nobody'", text)


if __name__ == "__main__":
    unittest.main()
