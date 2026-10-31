#!/usr/bin/env python3
"""bench.cli picks the speedup baseline from the dataset of the run instead of defaulting to ncnn's.

Run:  .venv/bin/python -m unittest bench.test_cli   (from the repo root)
"""
import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest import mock

from bench import cli
from contracts import BASELINE_AUTHORS, ISA_TABLE, baseline_author_for


def solution(name: str, dataset: str):
    return SimpleNamespace(name=name, dataset=SimpleNamespace(value=dataset))


TS = SimpleNamespace(solutions={
    "loop_001": [solution("reference_loop_001", "simd-loop"), solution("baseline-sve_loop_001", "simd-loop")],
    "gemm_q4_K_n1024_k2560": [solution("baseline-llamacpp-arm_gemm_q4_K_n1024_k2560", "llama.cpp")],
    "mixed": [solution("a", "ncnn"), solution("b", "llama.cpp")],
})


def resolve(*args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return cli.baseline_author_of(TS, *args, **kwargs)


class BaselineAuthorTest(unittest.TestCase):
    def test_an_explicit_author_wins_over_everything(self):
        self.assertEqual(resolve("my-baseline", "sve", dataset="ncnn"), "my-baseline")

    def test_bench_takes_the_dataset_of_the_named_solution(self):
        self.assertEqual(resolve(None, None, definition="gemm_q4_K_n1024_k2560",
                                 solution="baseline-llamacpp-arm_gemm_q4_K_n1024_k2560"),
                         BASELINE_AUTHORS["llama.cpp"])
        self.assertEqual(resolve(None, "sve", definition="loop_001", solution="reference_loop_001"),
                         baseline_author_for("simd-loop", "sve"))

    def test_the_isa_picks_the_tier_baseline_where_a_dataset_has_one(self):
        for isa in ISA_TABLE:
            self.assertEqual(resolve(None, isa, dataset="simd-loop"), baseline_author_for("simd-loop", isa))

    def test_collect_baselines_takes_the_dataset_from_the_definition_or_the_flag(self):
        self.assertEqual(resolve(None, None, definition="gemm_q4_K_n1024_k2560"), BASELINE_AUTHORS["llama.cpp"])
        self.assertEqual(resolve(None, None, dataset="kleidiai"), BASELINE_AUTHORS["kleidiai"])

    def test_it_exits_instead_of_guessing(self):
        with self.assertRaises(SystemExit) as no_dataset:
            resolve(None, None)
        self.assertIn("--baseline-author", str(no_dataset.exception))
        with self.assertRaises(SystemExit):
            resolve(None, None, definition="mixed")                       # two datasets: ambiguous
        with self.assertRaises(SystemExit) as no_solution:
            resolve(None, None, definition="loop_001", solution="nope")
        self.assertIn("nope", str(no_solution.exception))
        with self.assertRaises(SystemExit):
            resolve(None, None, dataset="raw")                            # no baseline author configured

    def test_the_subcommands_hand_the_derived_author_to_benchmark(self):
        bench_args = SimpleNamespace(root="r", definition="gemm_q4_K_n1024_k2560",
                                     solution="baseline-llamacpp-arm_gemm_q4_K_n1024_k2560",
                                     workload_axes=None, baseline_author=None, isa=None)
        collect_args = SimpleNamespace(root="r", definition=None, dataset="ncnn", baseline_author=None, isa=None)
        with mock.patch.object(cli.TraceSet, "from_path", return_value=TS), \
                mock.patch.object(cli, "Benchmark") as benchmark, contextlib.redirect_stdout(io.StringIO()):
            benchmark.return_value.bench.return_value = []
            cli.cmd_bench(bench_args)
            self.assertEqual(benchmark.call_args.args[1].baseline_author, BASELINE_AUTHORS["llama.cpp"])
            benchmark.return_value.collect_baselines.return_value = []
            cli.cmd_collect_baselines(collect_args)
            self.assertEqual(benchmark.call_args.args[1].baseline_author, BASELINE_AUTHORS["ncnn"])

    def test_the_parser_no_longer_defaults_to_ncnn(self):
        parser = cli.build_parser()
        self.assertIsNone(parser.parse_args(["bench", "--definition", "d", "--solution", "s"]).baseline_author)
        self.assertIsNone(parser.parse_args(["collect-baselines"]).baseline_author)
        args = parser.parse_args(["collect-baselines", "--dataset", "simd-loop", "--isa", "sve"])
        self.assertEqual((args.dataset, args.isa), ("simd-loop", "sve"))


if __name__ == "__main__":
    unittest.main()
