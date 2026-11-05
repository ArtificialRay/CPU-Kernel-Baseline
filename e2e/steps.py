"""The steps of `python -m e2e evaluate`, and the loop that runs them.

Each step is a function from the run context and the options to the arguments of the module
that does the work (kernel_set, deploy, measure, report). The loop:

  * skips a step whose artifact already exists (ctx.force redoes it);
  * only prints the command for a step that has to run on the Graviton box, unless it was
    asked for explicitly with `opts.steps` -- run on the box, that is how it gets run;
  * stops at the first failing step.

What comes before an e2e run (which operators the model uses, the definitions, real
activations, the calibrated SQNR gate) is not done here; see e2e/README.md.
"""
from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .context import E2EContext, Options
from .errors import StepFailed


def require(value, option: str, step: str):
    if value is None:
        raise StepFailed(f"the {step} step needs {option}")
    return value


def run_kernel_set(ctx: E2EContext, opts: Options):
    """Compile an author's submitted solutions into the override manifest, then check the guards."""
    from .kernel_set import build_kernel_set
    result = build_kernel_set(
        require(opts.author, "--author", "kernel_set"), ctx.run.manifest, isa=ctx.isa,
        solutions_root=opts.root / "solutions", only=opts.only, only_types=opts.only_types,
        only_roles=opts.only_roles, cxx=opts.cxx, march=opts.march, rows_abi=opts.rows_abi,
        expect_count=opts.expect_count, reference_author=opts.reference_author,
        expect_changed=opts.expect_changed, expect_same=opts.expect_same, forbid=opts.forbid)
    if result.guard_failures:
        raise StepFailed("\n  " + "\n  ".join(result.guard_failures))
    return result


def run_deploy(ctx: E2EContext, opts: Options) -> None:
    """llama.cpp tools tree, the model and (with agent_build) the override build, on the box."""
    from .deploy import deploy
    deploy(opts.label or f"e2e-{ctx.model}-{ctx.isa}", ctx.model, ctx.isa, opts.agent_build, opts.provision)


def run_measure(ctx: E2EContext, opts: Options) -> dict:
    """llama-bench (pp / tg) and llama-perplexity for stock / norepack / agent. Run it on the box."""
    from .measure import measure
    llama = Path(opts.llama_dir).expanduser()
    builds = {name: llama / f"build-{name}" for name in ("stock", "norepack", "agent")
              if (llama / f"build-{name}").is_dir()}
    if "stock" not in builds:
        raise StepFailed(f"no build-stock under {llama}; run the deploy step first")
    overrides = {"agent": ctx.run.manifest} if "agent" in builds and ctx.run.manifest.exists() else {}
    model = Path("~/models").expanduser() / ctx.model_config["file"]
    return measure(model, builds, overrides=overrides, threads=opts.threads, pp=opts.pp, tg=opts.tg,
                   reps=opts.reps, fa=opts.fa, ub=opts.ub,
                   perplexity=Path(opts.perplexity) if opts.perplexity else None,
                   ppl_chunks=opts.ppl_chunks, out=ctx.run.results)


def run_report(ctx: E2EContext, opts: Options) -> str:
    """Throughput, speedup and perplexity of the measured builds, as one table."""
    from .report import render_report
    text = render_report(ctx.run.results, opts.baseline)
    print(text)
    return text


@dataclass(frozen=True)
class Step:
    name: str
    run: Callable[[E2EContext, Options], object]
    box: bool = False                                           # has to run on the Graviton box
    outputs: Callable[[E2EContext], list] = lambda ctx: []      # artifacts that mean "already done"


STEPS = (
    Step("kernel_set", run_kernel_set, outputs=lambda ctx: [ctx.run.manifest]),
    Step("deploy", run_deploy),
    Step("measure", run_measure, box=True, outputs=lambda ctx: [ctx.run.results]),
    Step("report", run_report),
)


def run_steps(ctx: E2EContext, opts: Options) -> int:
    """After each agent run: compile the kernels, splice them into the model, measure."""
    known = [s.name for s in STEPS]
    wanted = opts.steps
    if wanted is not None:
        unknown = [n for n in wanted if n not in known]
        if unknown:
            raise SystemExit(f"evaluate has no step(s) {unknown}; steps: {known}")
    ctx.run.root.mkdir(parents=True, exist_ok=True)
    to_run_on_the_box: list[str] = []
    for step in STEPS:
        if wanted is not None and step.name not in wanted:
            continue
        done = step.outputs(ctx)
        if done and all(p.exists() for p in done) and not ctx.force:
            print(f"[e2e] {step.name}: skipped, {', '.join(map(str, done))} already there (--force to redo)")
            continue
        if step.box and wanted is None:
            args = ["python3", "-m", "e2e", "evaluate", "--model", ctx.model, "--isa", ctx.isa, "--steps", step.name]
            to_run_on_the_box.append(" ".join(map(shlex.quote, args)) + "   # plus that step's own options, see `python -m e2e evaluate -h`")
            print(f"[e2e] {step.name}: needs the Graviton box, not run from here -> see the command below")
            continue
        print(f"[e2e] {step.name}")
        try:
            step.run(ctx, opts)
        except StepFailed as e:
            print(f"[e2e] {step.name} failed: {e}", flush=True)
            return 1
    if to_run_on_the_box:
        print("\n[e2e] to run yourself on the box (from the repo checkout there):")
        for line in to_run_on_the_box:
            print(f"  {line}")
    return 0
