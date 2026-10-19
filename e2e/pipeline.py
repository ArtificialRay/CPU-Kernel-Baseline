"""The two composite commands, as plain ordered stage lists.

prepare   once per model:        inventory -> generate -> capture -> calibrate
evaluate  after each agent run:  kernel_set -> deploy -> measure -> report

Each stage turns the run context (plus a few user options) into the argument list of the
stage's own entrypoint, and says which artifact proves it already ran. A stage whose
artifact exists is skipped unless --force. Stages that need the Graviton box and are not
driven from here yet (capture, calibrate, measure) print the command to run instead.
"""
from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Callable

from .adapters.base import FrameworkAdapter
from .context import E2EContext


@dataclass
class Options:
    """The user-facing knobs of `prepare` / `evaluate`."""

    gguf_head: str | None = None          # prepare: a local GGUF header; downloaded when omitted
    author: str | None = None             # evaluate: whose bench-trace/solutions/<dataset>/<author>/ to compile
    label: str | None = None              # evaluate: provisioning/eval_config.json instance label
    agent_build: bool = False             # evaluate: also build the override-enabled tree on the box
    kernel_set_args: list[str] = field(default_factory=list)   # evaluate: --rows-abi, --reference-author, ...
    baseline: str = "stock"               # evaluate: build the report is relative to


@dataclass(frozen=True)
class Stage:
    name: str
    argv: Callable[[E2EContext, Options], list[str]]
    outputs: Callable[[E2EContext], list] = lambda ctx: []     # artifacts that mean "already done"
    manual: bool = False                                      # print the command instead of running it


def need(option: str, stage: str) -> str:
    raise SystemExit(f"the {stage} stage needs {option}")


def box_command(stage: str, argv: list[str]) -> str:
    return "python3 -m e2e " + " ".join([stage, *map(shlex.quote, argv)])


STAGES = {
    "inventory": Stage(
        "inventory",
        argv=lambda ctx, o: [str(ctx.run.gguf_head), "--json", str(ctx.run.inventory)],
        outputs=lambda ctx: [ctx.run.inventory],
    ),
    "generate": Stage(
        "generate",
        argv=lambda ctx, o: [str(ctx.run.inventory), "--model-tag", ctx.model],
    ),
    "capture": Stage(
        "capture",
        manual=True,
        argv=lambda ctx, o: ["--dump", "<ARMBENCH_DUMP_DIR contents>", "--root", "bench-trace", "--source", "<provenance text>"],
    ),
    "calibrate": Stage(
        "calibrate",
        manual=True,
        argv=lambda ctx, o: ["--tag", f"e2e:{ctx.model}"],
    ),
    "kernel_set": Stage(
        "kernel_set",
        argv=lambda ctx, o: ["--author", o.author or need("--author", "kernel_set"),
                             "--isa", ctx.isa, "--out", str(ctx.run.manifest), *o.kernel_set_args],
        outputs=lambda ctx: [ctx.run.manifest],
    ),
    "deploy": Stage(
        "deploy",
        argv=lambda ctx, o: ["--label", o.label or "<label>", "--model", ctx.model,
                             *(["--agent-build"] if o.agent_build else [])],
    ),
    "measure": Stage(
        "measure",
        manual=True,
        argv=lambda ctx, o: ["--model", f"~/models/{ctx.model_config['file']}",
                             "--build", "stock=<stock build dir>", "--build", "agent=<agent build dir>",
                             "--overrides", f"agent={ctx.run.manifest.name}", "--fa", "off",
                             "--out", ctx.run.results.name],
    ),
    "report": Stage(
        "report",
        argv=lambda ctx, o: [str(ctx.run.results), "--baseline", o.baseline],
    ),
}

GROUPS = {
    "prepare": ("inventory", "generate", "capture", "calibrate"),
    "evaluate": ("kernel_set", "deploy", "measure", "report"),
}


def select(group: str, start: str | None, stop: str | None) -> list[str]:
    names = list(GROUPS[group])
    for bound in (start, stop):
        if bound is not None and bound not in names:
            raise SystemExit(f"{bound!r} is not a stage of {group!r}; stages: {names}")
    lo = names.index(start) if start else 0
    hi = names.index(stop) if stop else len(names) - 1
    return names[lo:hi + 1]


def run_group(
    group: str, ctx: E2EContext, adapter: FrameworkAdapter, options: Options,
    start: str | None = None, stop: str | None = None,
) -> int:
    ctx.run.root.mkdir(parents=True, exist_ok=True)
    manual: list[str] = []
    for name in select(group, start, stop):
        stage = STAGES[name]
        done = stage.outputs(ctx)
        if done and all(p.exists() for p in done) and not ctx.force:
            print(f"[e2e] {name}: skipped, {', '.join(map(str, done))} already there (--force to redo)")
            continue
        if name == "inventory" and not ctx.run.gguf_head.exists() and not options.gguf_head:
            print(f"[e2e] inventory: fetching the GGUF header of {ctx.model} -> {ctx.run.gguf_head}")
            adapter.fetch_header(ctx.model_config, ctx.run.gguf_head)
        argv = stage.argv(ctx, options)
        if name == "inventory" and options.gguf_head:
            argv[0] = options.gguf_head
        if name == "generate" and ctx.force:
            argv.append("--force")
        if stage.manual:
            manual.append(box_command(name, argv))
            print(f"[e2e] {name}: runs on the box, not driven from here yet -> see the command below")
            continue
        print(f"[e2e] {name}: {box_command(name, argv)}")
        rc = adapter.entrypoint(name)(argv)
        if rc:
            print(f"[e2e] {name} failed (exit {rc}); stopping", flush=True)
            return rc
    if manual:
        print("\n[e2e] to run yourself on the box:")
        for line in manual:
            print(f"  {line}")
    return 0
