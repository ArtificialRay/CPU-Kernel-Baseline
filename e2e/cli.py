"""python -m e2e prepare|evaluate ...

  python -m e2e prepare  --model qwen3.5-4b [--gguf-head FILE] [--force]
  python -m e2e evaluate --model qwen3.5-4b --author AUTHOR --label BOX [--agent-build] [--rows-abi]

prepare (once per model)       inventory -> calibrate
evaluate (after each agent run) kernel_set -> deploy -> measure -> report

Steps that need the Graviton box (calibrate, measure) are printed, not run, unless
named with --steps: on the box, `python -m e2e evaluate --model M --steps measure ...` runs
just that one. Everything else runs where the command runs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .adapters.registry import resolve_adapter
from .context import DEFAULT_FRAMEWORK, E2EContext, Options

COMMANDS = ("prepare", "evaluate")


def build_parser(command: str, framework: str = DEFAULT_FRAMEWORK) -> argparse.ArgumentParser:
    adapter = resolve_adapter(framework)
    steps = adapter.prepare_steps if command == "prepare" else adapter.evaluate_steps
    names = [s.name for s in steps]
    p = argparse.ArgumentParser(prog=f"e2e {command}", description=" -> ".join(names))
    p.add_argument("--model", required=True, help="key in config/e2e_models.json")
    p.add_argument("--isa", default="sve2")
    p.add_argument("--run-dir", default=None, help="default: e2e-runs/<framework>-<model>-<isa>")
    p.add_argument("--steps", default=None,
                   help=f"comma-separated subset of {','.join(names)}; named steps run even when they need the box")
    p.add_argument("--force", action="store_true", help="redo steps whose artifact already exists")
    p.add_argument("--root", type=Path, default=Options().root, help="warehouse root (default: bench-trace/)")
    if command == "prepare":
        p.add_argument("--gguf-head", default=None, help="a local GGUF header (default: download the first 64 MiB)")
    else:
        k = p.add_argument_group("kernel_set")
        k.add_argument("--author", default=None, help="whose bench-trace/solutions/<dataset>/<author>/ to compile")
        k.add_argument("--rows-abi", action="store_true", help="let the override split rows across ggml's threads")
        k.add_argument("--only", nargs="*", default=None)
        k.add_argument("--only-types", nargs="*", default=None)
        k.add_argument("--only-roles", nargs="*", default=None)
        k.add_argument("--reference-author", default=None, help="guards compare against this author's solutions")
        k.add_argument("--expect-count", type=int, default=None)
        k.add_argument("--expect-changed", nargs="*", default=[])
        k.add_argument("--expect-same", nargs="*", default=[])
        k.add_argument("--forbid", nargs="*", default=[])
        k.add_argument("--cxx", default="clang++-18")
        k.add_argument("--march", default=None)
        d = p.add_argument_group("deploy")
        d.add_argument("--label", default=None, help="instance label in provisioning/eval_config.json")
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
    framework = DEFAULT_FRAMEWORK
    if argv[:1] == ["--framework"] and len(argv) >= 2:
        framework, argv = argv[1], argv[2:]
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in COMMANDS:
        print(__doc__, file=sys.stderr if argv and argv[0] not in ("-h", "--help") else sys.stdout)
        return 0 if not argv or argv[0] in ("-h", "--help") else 2
    command = argv[0]
    args = build_parser(command, framework).parse_args(argv[1:])
    fields = {k: v for k, v in vars(args).items() if k in Options.__dataclass_fields__}
    fields["steps"] = args.steps.split(",") if args.steps else None
    options = Options(**fields)
    ctx = E2EContext.create(framework, args.model, args.isa, args.run_dir, args.force)
    return getattr(resolve_adapter(framework), command)(ctx, options)


if __name__ == "__main__":
    raise SystemExit(main())
