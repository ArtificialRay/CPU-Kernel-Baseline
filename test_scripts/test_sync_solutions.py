#!/usr/bin/env python3
"""Tests for bench_fleet.py's --sync-solutions helpers (no instance, no network).

Run:  .venv/bin/python -m unittest test_scripts.test_sync_solutions   (from the repo root)
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "test_scripts"))

import bench_fleet  # noqa: E402
from skills.launch import launch_session  # noqa: E402

# A definition whose op_type differs from its definitions/ subdirectory name.
DEFINITION = "loop_027"
DATASET = "simd-loop"
AUTHOR = "unit-author"


class FakeTarget:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls = []

    def rsync_from(self, remote_path, local_dir, excludes=None):
        self.calls.append((remote_path, Path(local_dir), list(excludes or [])))
        if self.fail:
            raise RuntimeError("rsync exited 23")


def write_trajectory(results_dir: Path, name: str, records: list[dict]) -> None:
    path = results_dir / name / "trajectory.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


class SyncSolutionsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.saved = bench_fleet.LOCAL_SOLUTIONS_DIR
        bench_fleet.LOCAL_SOLUTIONS_DIR = self.tmp / "solutions"
        self.results = self.tmp / "agent-runs"

    def tearDown(self):
        bench_fleet.LOCAL_SOLUTIONS_DIR = self.saved

    def test_op_type_comes_from_the_definition_json(self):
        self.assertEqual(bench_fleet.definition_op_type(DEFINITION), DEFINITION)
        self.assertIsNone(bench_fleet.definition_op_type("no_such_definition"))

    def test_pulls_exactly_the_submitted_file(self):
        target = FakeTarget()
        ok = bench_fleet.sync_job_solution(target, DATASET, AUTHOR, DEFINITION, "~/arm-bench")
        self.assertTrue(ok)
        (remote, local_dir, excludes), = target.calls
        self.assertEqual(
            remote, f"~/arm-bench/bench-trace/solutions/{DATASET}/{AUTHOR}/{DEFINITION}/{DEFINITION}.json",
        )
        self.assertEqual(local_dir, bench_fleet.LOCAL_SOLUTIONS_DIR / DATASET / AUTHOR / DEFINITION)
        self.assertEqual(excludes, launch_session.SOLUTION_INTERMEDIATE_GLOBS)
        self.assertIn("*_current.json", excludes)
        self.assertIn("*_best.json", excludes)

    def test_failure_is_reported_not_raised(self):
        self.assertFalse(bench_fleet.sync_job_solution(
            FakeTarget(fail=True), DATASET, AUTHOR, DEFINITION, "~/arm-bench"))
        self.assertFalse(bench_fleet.sync_job_solution(
            FakeTarget(), DATASET, AUTHOR, "no_such_definition", "~/arm-bench"))

    def test_submitted_ok_needs_a_passed_submit(self):
        write_trajectory(self.results, "passed", [
            {"tool": "compile"}, {"tool": "evaluate"}, {"tool": "submit", "metrics": {"status": "PASSED"}}])
        write_trajectory(self.results, "nothing_evaluated", [
            {"tool": "compile"}, {"tool": "submit", "metrics": {"status": "COMPILE_ERROR"}}])
        write_trajectory(self.results, "no_submit", [{"tool": "compile"}, {"tool": "evaluate"}])
        self.assertTrue(bench_fleet.submitted_ok(self.results, "passed"))
        self.assertFalse(bench_fleet.submitted_ok(self.results, "nothing_evaluated"))
        self.assertFalse(bench_fleet.submitted_ok(self.results, "no_submit"))
        self.assertFalse(bench_fleet.submitted_ok(self.results, "no_trajectory"))

    def test_find_unsynced_flags_only_submitted_and_missing(self):
        submitted = [{"tool": "evaluate"}, {"tool": "submit", "metrics": {"status": "PASSED"}}]
        for name in ("loop_027", "loop_029", "loop_126"):
            write_trajectory(self.results, name, submitted)
        write_trajectory(self.results, "loop_101", [{"tool": "submit", "metrics": {"status": "COMPILE_ERROR"}}])
        # loop_027 present locally, loop_029 and loop_126 missing, loop_101 never had a solution
        present = bench_fleet.local_solution_path(DATASET, AUTHOR, "loop_027")
        present.parent.mkdir(parents=True)
        present.write_text("{}")
        missing = bench_fleet.find_unsynced_solutions(
            DATASET, AUTHOR, ["loop_027", "loop_029", "loop_126", "loop_101"], self.results)
        self.assertEqual(missing, ["loop_029", "loop_126"])

    def test_report_writes_and_clears_the_marker(self):
        marker = self.results / bench_fleet.UNSYNCED_FILENAME
        bench_fleet.report_unsynced_solutions(self.results, {DATASET: ["loop_029"]})
        self.assertEqual(marker.read_text(), f"{DATASET}\tloop_029\n")
        bench_fleet.report_unsynced_solutions(self.results, {})
        self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
