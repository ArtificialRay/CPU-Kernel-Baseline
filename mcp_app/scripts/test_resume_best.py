#!/usr/bin/env python3
"""What happens to the persisted best solution when the MCP server restarts mid-run.

KernelSession keeps `best_compile` in memory (agent_tools/base.py), seeded to None for
every definition in a new process; only the turn/version counters are recovered from the
trajectory on disk. So the first PASSED evaluate after a restart becomes "the best" and
auto-submits, overwriting `<definition>.json` even if it is slower than what was
submitted before the restart. The agent-side mitigation is the SKILL.md instruction to
check_progress(), recompile the returned best_code and evaluate it first.

No hardware, no compiler: ops.compile_kernel / ops.evaluate_kernel are stubbed.

Run:  .venv/bin/python -m unittest mcp_app.scripts.test_resume_best   (from the repo root)
"""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from bench.data.solution import Solution, SolutionSpec, SourceFile, SupportedDatasets
from mcp_app.agent_tools import baseline_readiness, ops
from mcp_app.agent_tools.base import KernelSession

DEFINITION = "gemm_ggml_q4_K_n1024_k2560"
AUTHOR = "unit-author"


class FakeTraceSet:
    def __init__(self, root: Path):
        self.root = root
        self.definitions = {DEFINITION: SimpleNamespace(name=DEFINITION, op_type="gemm")}
        self.solutions = {DEFINITION: [SimpleNamespace(dataset=SupportedDatasets.LLAMA_CPP)]}

    def get_baseline_solution(self, name, author):
        return None

    def add_traces(self, traces):
        pass


class FakeSession(KernelSession):
    dataset = "llama.cpp"

    def make_solution(self, code):
        return Solution(
            name=f"{self._author}_{DEFINITION}", definition=DEFINITION,
            dataset=SupportedDatasets.LLAMA_CPP, author=self._author,
            spec=SolutionSpec(target_hardware=["aarch64"], entry_point="kernel.cpp::f"),
            sources=[SourceFile(path="kernel.cpp", content=code)],
        )

    @classmethod
    def tool_schemas(cls):
        return []


class ResumeBestTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.speedups = []          # consumed by the evaluate stub, one per evaluate()
        self.n_compiles = 0
        patches = [
            mock.patch.object(ops, "compile_kernel", side_effect=self.fake_compile),
            mock.patch.object(ops, "evaluate_kernel", side_effect=self.fake_evaluate),
            mock.patch.object(baseline_readiness, "ensure_baseline_collected"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def fake_compile(self, definition, solution):
        self.n_compiles += 1
        build = self.tmp / "build" / str(self.n_compiles)
        build.mkdir(parents=True)
        return {"status": "OK", "so_path": str(build / "k.so")}

    def fake_evaluate(self, trace_set, definition, so_path, solution_name, cfg):
        s = self.speedups.pop(0)
        return {
            "status": "PASSED", "correctness": {}, "traces": [],
            "performance": {"time_speedup_geomean": s, "cycle_speedup_geomean": s},
        }

    def new_process(self) -> FakeSession:
        """A fresh server process on the same run_dir / bench-trace root."""
        return FakeSession(
            FakeTraceSet(self.tmp / "bench-trace"), AUTHOR,
            SimpleNamespace(baseline_author="baseline-llamacpp-arm"),
            self.tmp / "agent-runs", "sve2",
        )

    def compile_and_evaluate(self, sess: FakeSession, code: str, speedup: float) -> None:
        self.speedups.append(speedup)
        r = sess.compile(DEFINITION, code)
        self.assertEqual(r["status"], "OK", r)
        sess.evaluate(DEFINITION, r["version"])

    def persisted_kernel(self) -> str:
        path = self.tmp / "bench-trace" / "solutions" / "llama.cpp" / AUTHOR / "gemm" / f"{DEFINITION}.json"
        sources = json.loads(path.read_text())["sources"]
        return next(s["content"] for s in sources if s["path"] == "kernel.cpp")

    def test_worse_result_in_the_same_process_does_not_replace_the_best(self):
        sess = self.new_process()
        self.compile_and_evaluate(sess, "fast", 1.30)
        self.compile_and_evaluate(sess, "slow", 1.10)
        self.assertEqual(self.persisted_kernel(), "fast")

    @unittest.expectedFailure
    def test_restart_does_not_let_a_slower_kernel_replace_the_best(self):
        """The invariant we want, and a known gap: best_compile starts at None again after
        a restart, so the persisted <def>.json becomes the slower kernel. Left out of the
        server on purpose (only the agent-side check_progress recovery covers it); when
        the server learns to seed its best from the trajectory, this starts passing and
        unittest reports an unexpected success — drop the decorator then."""
        first = self.new_process()
        self.compile_and_evaluate(first, "fast", 1.30)
        second = self.new_process()
        self.compile_and_evaluate(second, "slow", 1.10)
        self.assertEqual(self.persisted_kernel(), "fast")

    def test_check_progress_then_recompile_restores_the_baseline_to_beat(self):
        """The SKILL.md recovery path: compile + evaluate best_code first, then optimize."""
        first = self.new_process()
        self.compile_and_evaluate(first, "fast", 1.30)
        second = self.new_process()
        progress = second.check_progress(DEFINITION)
        self.assertTrue(progress["has_prior_progress"])
        self.assertEqual(progress["best_code"], "fast")
        self.compile_and_evaluate(second, progress["best_code"], 1.29)   # re-measured, noise
        self.compile_and_evaluate(second, "slow", 1.10)
        self.assertEqual(self.persisted_kernel(), "fast")


if __name__ == "__main__":
    unittest.main()
