#!/usr/bin/env python3
"""The e2e stage list, the CLI dispatch and the skip/--force rules, against a fake
framework adapter (no model, no network, no box).

Run:  .venv/bin/python -m unittest e2e.test_pipeline   (from the repo root)
"""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from e2e import cli, pipeline
from e2e.adapters.base import STAGE_NAMES, Entrypoint, FrameworkAdapter
from e2e.adapters.registry import register_adapter, resolve_adapter
from e2e.context import E2EContext

MODEL = "qwen3.5-4b"      # any key of config/e2e_models.json


class NullAdapter(FrameworkAdapter):
    """Records every stage call; stages that have an artifact write it, like the real ones."""

    name = "null"

    def __init__(self):
        self.calls: list[tuple[str, list[str]]] = []
        self.headers: list[Path] = []

    def entrypoints(self) -> dict[str, Entrypoint]:
        return {stage: self.make(stage) for stage in STAGE_NAMES}

    def make(self, stage: str) -> Entrypoint:
        def run(argv: list[str]) -> int:
            self.calls.append((stage, argv))
            if stage == "inventory":
                Path(argv[argv.index("--json") + 1]).write_text("{}")
            if stage == "kernel_set":
                Path(argv[argv.index("--out") + 1]).write_text("{}")
            return 1 if "--fail" in argv else 0
        return run

    def fetch_header(self, model_config: dict, dest: Path) -> Path:
        self.headers.append(dest)
        dest.write_bytes(b"GGUF")
        return dest


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.adapter = NullAdapter()
        register_adapter("null", lambda: self.adapter)
        self.ctx = E2EContext.create("null", MODEL, "sve2", Path(tempfile.mkdtemp()))
        self.out = io.StringIO()

    def run_group(self, group, options=None, **kw):
        with contextlib.redirect_stdout(self.out):
            return pipeline.run_group(group, self.ctx, self.adapter, options or pipeline.Options(), **kw)

    def stages_run(self):
        return [name for name, _ in self.adapter.calls]

    def test_select_ranges(self):
        self.assertEqual(pipeline.select("evaluate", None, None), ["kernel_set", "deploy", "measure", "report"])
        self.assertEqual(pipeline.select("evaluate", "deploy", "measure"), ["deploy", "measure"])
        with self.assertRaises(SystemExit):
            pipeline.select("evaluate", "inventory", None)

    def test_evaluate_runs_local_stages_and_prints_the_box_command(self):
        options = pipeline.Options(label="box1", agent_build=True, kernel_set_args=["--solutions", "x"])
        self.assertEqual(self.run_group("evaluate", options), 0)
        self.assertEqual(self.stages_run(), ["kernel_set", "deploy", "report"])     # measure is manual
        kernel_set = dict(self.adapter.calls)["kernel_set"]
        self.assertEqual(kernel_set[:4], ["--isa", "sve2", "--out", str(self.ctx.run.manifest)])
        self.assertEqual(kernel_set[4:], ["--solutions", "x"])
        self.assertEqual(dict(self.adapter.calls)["deploy"],
                         ["--label", "box1", "--model", MODEL, "--agent-build"])
        self.assertIn("python3 -m e2e measure", self.out.getvalue())
        self.assertIn("to run yourself on the box", self.out.getvalue())

    def test_existing_artifact_is_skipped_unless_forced(self):
        self.run_group("evaluate", stop="kernel_set")
        self.run_group("evaluate", stop="kernel_set")
        self.assertEqual(self.stages_run(), ["kernel_set"])                          # second run skipped
        self.ctx = E2EContext.create("null", MODEL, "sve2", self.ctx.run.root, force=True)
        self.run_group("evaluate", stop="kernel_set")
        self.assertEqual(self.stages_run(), ["kernel_set", "kernel_set"])

    def test_a_failing_stage_stops_the_pipeline(self):
        options = pipeline.Options(kernel_set_args=["--fail"])
        self.assertEqual(self.run_group("evaluate", options), 1)
        self.assertEqual(self.stages_run(), ["kernel_set"])

    def test_prepare_fetches_the_header_unless_one_is_given(self):
        self.run_group("prepare", stop="inventory")
        self.assertEqual(self.adapter.headers, [self.ctx.run.gguf_head])
        inventory = dict(self.adapter.calls)["inventory"]
        self.assertEqual(inventory, [str(self.ctx.run.gguf_head), "--json", str(self.ctx.run.inventory)])

        other = E2EContext.create("null", MODEL, "sve2", Path(tempfile.mkdtemp()))
        self.ctx, self.adapter.calls, self.adapter.headers = other, [], []
        self.run_group("prepare", pipeline.Options(gguf_head="/data/head.bin"), stop="inventory")
        self.assertEqual(self.adapter.headers, [])
        self.assertEqual(dict(self.adapter.calls)["inventory"][0], "/data/head.bin")

    def test_cli_bare_stage_passes_its_arguments_through(self):
        with contextlib.redirect_stdout(self.out):
            rc = cli.main(["--framework", "null", "kernel-set", "--runs", "r", "--out", "m.json"])
        self.assertEqual(rc, 0)
        self.assertEqual(self.adapter.calls, [("kernel_set", ["--runs", "r", "--out", "m.json"])])

    def test_cli_composite_builds_the_context(self):
        run_dir = str(Path(tempfile.mkdtemp()) / "run")
        with contextlib.redirect_stdout(self.out):
            rc = cli.main(["--framework", "null", "evaluate", "--model", MODEL, "--label", "b",
                           "--run-dir", run_dir, "--to", "kernel_set", "--kernel-set-args", "--solutions a"])
        self.assertEqual(rc, 0)
        self.assertEqual(self.stages_run(), ["kernel_set"])
        self.assertEqual(self.adapter.calls[0][1][-2:], ["--solutions", "a"])

    def test_unknown_command_and_unknown_adapter(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(["--framework", "null", "nope"]), 2)
        with self.assertRaises(SystemExit):
            resolve_adapter("no-such-framework")

    def test_llamacpp_adapter_lists_every_stage_without_importing_them(self):
        stages = resolve_adapter("llamacpp").entrypoints()
        self.assertEqual(sorted(stages), sorted(STAGE_NAMES))


if __name__ == "__main__":
    unittest.main()
