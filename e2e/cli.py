"""python -m e2e evaluate ...

  python -m e2e evaluate --model qwen3.5-4b --author AUTHOR [--label BOX] [--agent-build] [--rows-abi]

evaluate (after each agent run) kernel_set -> deploy -> measure -> report

Before the first run the model's definitions, real activations and calibrated SQNR gate are
made by the scripts in scripts/e2e (see e2e/README.md); this command starts from the agents' submitted solutions.

The step that needs the Graviton box (measure) is printed, not run, unless
named with --steps: on the box, `python -m e2e evaluate --model M --steps measure ...` runs
just that one. Everything else runs where the command runs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .context import DEFAULT_FRAMEWORK, E2EContext, Options
from .steps import STEPS, run_steps

COMMANDS = ("evaluate",)


def build_parser() -> argparse.ArgumentParser:
    names = [s.name for s in STEPS]
    p = argparse.ArgumentParser(prog="e2e evaluate", description=" -> ".join(names))
    p.add_argument("--model", required=True, help="key in config/e2e_models.json")
    p.add_argument("--isa", default="sve2")
    p.add_argument("--run-dir", default=None, help="default: e2e-runs/<framework>-<model>-<isa>")
    p.add_argument("--steps", default=None,
                   help=f"comma-separated subset of {','.join(names)}; named steps run even when they need the box")
    p.add_argument("--force", action="store_true", help="redo steps whose artifact already exists")
    p.add_argument("--root", type=Path, default=Options().root, help="warehouse root (default: bench-trace/)")
    k = p.add_argument_group("kernel_set")
    k.add_argument("--author", default=None, help="whose bench-trace/solutions/<dataset>/<author>/ to compile")
    k.add_argument("--rows-abi", action="store_true", help="let the override split rows across ggml's threads")
    k.add_argument("--kernel", default="all", help="'all' (every definition tagged e2e:<model>) or one definition name")
    k.add_argument("--reference-author", default=None, help="guards compare against this author's solutions")
    k.add_argument("--expect-count", type=int, default=None)
    k.add_argument("--expect-changed", nargs="*", default=[])
    k.add_argument("--expect-same", nargs="*", default=[])
    k.add_argument("--forbid", nargs="*", default=[])
    k.add_argument("--cxx", default="clang++-18")
    k.add_argument("--march", default=None)
    d = p.add_argument_group("deploy")
    d.add_argument("--label", default=None, help="instance label in provisioning/eval_config.json (default: e2e-<model>-<isa>)")
    d.add_argument("--provision", action="store_true", help="create the box if none is registered under the label (spends money)")
    d.add_argument("--agent-build", action="store_true", help="also build the override-enabled tree")
    m = p.add_argument_group("measure (on the box) / report")
    m.add_argument("--llama-dir", default="~/llama.cpp-e2e")
    m.add_argument("--threads", nargs="+", type=int, default=[4])
    m.add_argument("--pp", type=int, default=512)
    m.add_argument("--tg", type=int, default=128)
    m.add_argument("--reps", type=int, default=5)
    m.add_argument("--fa", choices=["on", "off"], default="off", help="flash attention; never left to auto")
    m.add_argument("--ub", type=int, default=None)
    m.add_argument("--perplexity", default=None, help="text file for llama-perplexity (the quality check)")
    m.add_argument("--ppl-chunks", type=int, default=64)
    m.add_argument("--baseline", default="stock", help="report: the build speedups are relative to")
    return p


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)   # never lose progress lines to a pipe buffer
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in COMMANDS:
        print(__doc__, file=sys.stderr if argv and argv[0] not in ("-h", "--help") else sys.stdout)
        return 0 if not argv or argv[0] in ("-h", "--help") else 2
    args = build_parser().parse_args(argv[1:])
    fields = {k: v for k, v in vars(args).items() if k in Options.__dataclass_fields__}
    fields["steps"] = args.steps.split(",") if args.steps else None
    ctx = E2EContext.create(DEFAULT_FRAMEWORK, args.model, args.isa, args.run_dir, args.force)
    return run_steps(ctx, Options(**fields))


if __name__ == "__main__":
    raise SystemExit(main())
