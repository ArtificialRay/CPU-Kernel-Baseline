#!/usr/bin/env python3
"""The two commands, the step loop and the CLI, against a fake framework (no model, no network, no box).

Run:  .venv/bin/python -m unittest e2e.test_pipeline   (from the repo root)
"""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from e2e import cli
from e2e.adapters.base import FrameworkAdapter, Step
from e2e.adapters.registry import register_adapter, resolve_adapter
from e2e.context import E2EContext, Options
from e2e.errors import StepFailed

MODEL = "qwen3.5-4b"      # any key of config/e2e_models.json


class NullAdapter(FrameworkAdapter):
    """Records every step; steps that have an artifact write it, like the real ones."""

    name = "null"
    prepare_steps = (
        Step("inventory", outputs=lambda ctx: [ctx.run.inventory]),
        Step("generate"),
        Step("capture", box=True),
    )
    evaluate_steps = (
        Step("kernel_set", outputs=lambda ctx: [ctx.run.manifest]),
        Step("deploy"),
        Step("measure", box=True),
        Step("report"),
    )

    def __init__(self):
        self.calls: list[tuple[str, Options]] = []
        self.headers: list[Path] = []
        self.fail_on: str | None = None

    def run_step(self, step, ctx, opts):
        self.calls.append((step, opts))
        if step == self.fail_on:
            raise StepFailed("boom")
        if step == "inventory":
            ctx.run.inventory.write_text("{}")
        if step == "kernel_set":
            ctx.run.manifest.write_text("{}")

    def fetch_header(self, model_config, dest):
        self.headers.append(dest)
        return dest

    def stages_run(self):
        return [name for name, _ in self.calls]


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.adapter = NullAdapter()
        register_adapter("null", lambda: self.adapter)
        self.ctx = E2EContext.create("null", MODEL, "sve2", Path(tempfile.mkdtemp()))
        self.out = io.StringIO()

    def run_cmd(self, command, options=None, ctx=None):
        with contextlib.redirect_stdout(self.out):
            return getattr(self.adapter, command)(ctx or self.ctx, options or Options())

    def test_evaluate_runs_the_local_steps_and_prints_the_box_ones(self):
        self.assertEqual(self.run_cmd("evaluate"), 0)
        self.assertEqual(self.adapter.stages_run(), ["kernel_set", "deploy", "report"])    # measure needs the box
        text = self.out.getvalue()
        self.assertIn("python3 -m e2e evaluate --model qwen3.5-4b --isa sve2 --steps measure", text)
        self.assertIn("to run yourself on the box", text)

    def test_a_named_step_runs_even_when_it_needs_the_box(self):
        self.assertEqual(self.run_cmd("evaluate", Options(steps=["measure"])), 0)
        self.assertEqual(self.adapter.stages_run(), ["measure"])
        self.assertNotIn("to run yourself", self.out.getvalue())

    def test_unknown_step_is_an_error(self):
        with contextlib.redirect_stdout(self.out), self.assertRaises(SystemExit):
            self.adapter.evaluate(self.ctx, Options(steps=["nope"]))

    def test_existing_artifact_is_skipped_unless_forced(self):
        options = Options(steps=["kernel_set"])
        self.run_cmd("evaluate", options)
        self.run_cmd("evaluate", options)
        self.assertEqual(self.adapter.stages_run(), ["kernel_set"])                         # second run skipped
        forced = E2EContext.create("null", MODEL, "sve2", self.ctx.run.root, force=True)
        self.run_cmd("evaluate", options, forced)
        self.assertEqual(self.adapter.stages_run(), ["kernel_set", "kernel_set"])

    def test_a_failing_step_stops_the_command(self):
        self.adapter.fail_on = "kernel_set"
        self.assertEqual(self.run_cmd("evaluate"), 1)
        self.assertEqual(self.adapter.stages_run(), ["kernel_set"])
        self.assertIn("kernel_set failed: boom", self.out.getvalue())

    def test_prepare_runs_inventory_then_generate(self):
        self.assertEqual(self.run_cmd("prepare"), 0)
        self.assertEqual(self.adapter.stages_run(), ["inventory", "generate"])               # capture needs the box

    def test_cli_builds_the_options_from_the_flags(self):
        run_dir = str(Path(tempfile.mkdtemp()) / "run")
        with contextlib.redirect_stdout(self.out):
            rc = cli.main(["--framework", "null", "evaluate", "--model", MODEL, "--author", "who",
                           "--label", "box", "--rows-abi", "--steps", "kernel_set,report", "--run-dir", run_dir,
                           "--expect-count", "3", "--threads", "4", "8", "--perplexity", "wiki.raw"])
        self.assertEqual(rc, 0)
        (_, opts), _ = self.adapter.calls
        self.assertEqual((opts.author, opts.label, opts.rows_abi, opts.expect_count, opts.threads, opts.perplexity),
                         ("who", "box", True, 3, [4, 8], "wiki.raw"))
        self.assertEqual(opts.steps, ["kernel_set", "report"])

    def test_only_prepare_and_evaluate_exist(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(["kernel_set", "--out", "x"]), 2)
        with self.assertRaises(SystemExit):
            resolve_adapter("no-such-framework")


if __name__ == "__main__":
    unittest.main()
