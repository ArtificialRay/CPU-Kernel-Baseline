"""The framework-specific half of the e2e pipeline, and the loop that drives it.

A framework is one class. Its `prepare_steps` / `evaluate_steps` name the steps of the two
commands, and `run_step(name, ctx, opts)` does one of them. The loop here is the same for
every framework:

  * a step whose artifact already exists is skipped (ctx.force redoes it);
  * a step that has to run on the Graviton box is only printed as a command unless it was
    asked for explicitly with `opts.steps` -- run on the box, that is how it gets run;
  * the first failing step stops the command.

The interface is deliberately this coarse: llama.cpp is the only framework so far, and a
finer split drawn from one implementation would be a guess. Split it when a second
framework shows what differs.
"""
from __future__ import annotations

import shlex
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..context import E2EContext, Options
from ..errors import StepFailed


@dataclass(frozen=True)
class Step:
    name: str
    box: bool = False                                           # has to run on the Graviton box
    outputs: Callable[[E2EContext], list] = lambda ctx: []      # artifacts that mean "already done"


class FrameworkAdapter(ABC):
    name: str
    prepare_steps: tuple[Step, ...]
    evaluate_steps: tuple[Step, ...]

    @abstractmethod
    def run_step(self, step: str, ctx: E2EContext, opts: Options) -> None:
        """Do one step; raise StepFailed when it cannot continue."""

    @abstractmethod
    def fetch_header(self, model_config: dict, dest: Path) -> Path:
        """Download just enough of the model file for the inventory step."""

    def prepare(self, ctx: E2EContext, opts: Options) -> int:
        """Once per model: define its kernel set and calibrate the correctness gate."""
        return self.run_group("prepare", self.prepare_steps, ctx, opts)

    def evaluate(self, ctx: E2EContext, opts: Options) -> int:
        """After each agent run: compile the kernels, splice them into the model, measure."""
        return self.run_group("evaluate", self.evaluate_steps, ctx, opts)

    def run_group(self, group: str, steps: tuple[Step, ...], ctx: E2EContext, opts: Options) -> int:
        known = [s.name for s in steps]
        wanted = opts.steps
        if wanted is not None:
            unknown = [n for n in wanted if n not in known]
            if unknown:
                raise SystemExit(f"{group} has no step(s) {unknown}; steps: {known}")
        ctx.run.root.mkdir(parents=True, exist_ok=True)
        to_run_on_the_box: list[str] = []
        for step in steps:
            if wanted is not None and step.name not in wanted:
                continue
            done = step.outputs(ctx)
            if done and all(p.exists() for p in done) and not ctx.force:
                print(f"[e2e] {step.name}: skipped, {', '.join(map(str, done))} already there (--force to redo)")
                continue
            if step.box and wanted is None:
                to_run_on_the_box.append(self.box_command(group, step.name, ctx))
                print(f"[e2e] {step.name}: needs the Graviton box, not run from here -> see the command below")
                continue
            print(f"[e2e] {step.name}")
            try:
                self.run_step(step.name, ctx, opts)
            except StepFailed as e:
                print(f"[e2e] {step.name} failed: {e}", flush=True)
                return 1
        if to_run_on_the_box:
            print("\n[e2e] to run yourself on the box (from the repo checkout there):")
            for line in to_run_on_the_box:
                print(f"  {line}")
        return 0

    def box_command(self, group: str, step: str, ctx: E2EContext) -> str:
        args = ["python3", "-m", "e2e", group, "--model", ctx.model, "--isa", ctx.isa, "--steps", step]
        return " ".join(map(shlex.quote, args)) + "   # plus that step's own options, see `python -m e2e " + group + " -h`"
