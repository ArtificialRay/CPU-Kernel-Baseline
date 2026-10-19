"""python -m e2e ...

  python -m e2e prepare  --model qwen3.5-4b [--gguf-head FILE] [--from STAGE] [--to STAGE] [--force]
  python -m e2e evaluate --model qwen3.5-4b --label BOX [--agent-build] [--kernel-set-args "--runs DIR --rows-abi"]
  python -m e2e [--framework NAME] <stage> <that stage's own arguments>      # e.g. kernel_set, measure

A bare stage name runs exactly the program the old scripts/e2e/<script>.py was.
"""
from __future__ import annotations

import argparse
import shlex
import sys

from .adapters.base import STAGE_NAMES
from .adapters.registry import resolve_adapter
from .context import DEFAULT_FRAMEWORK, E2EContext
from .pipeline import GROUPS, Options, run_group


def composite_parser(group: str) -> argparse.ArgumentParser:
    stages = GROUPS[group]
    p = argparse.ArgumentParser(prog=f"e2e {group}", description=f"{' -> '.join(stages)}")
    p.add_argument("--model", required=True, help="key in config/e2e_models.json")
    p.add_argument("--isa", default="sve2")
    p.add_argument("--run-dir", default=None, help="default: e2e-runs/<framework>-<model>-<isa>")
    p.add_argument("--from", dest="start", choices=stages, default=None)
    p.add_argument("--to", dest="stop", choices=stages, default=None)
    p.add_argument("--force", action="store_true", help="redo stages whose artifact already exists")
    if group == "prepare":
        p.add_argument("--gguf-head", default=None, help="a local GGUF header (default: download the first 64 MiB)")
    else:
        p.add_argument("--label", default=None, help="instance label in provisioning/eval_config.json")
        p.add_argument("--agent-build", action="store_true")
        p.add_argument("--kernel-set-args", default="", help='extra arguments for kernel_set, e.g. "--runs DIR --rows-abi"')
        p.add_argument("--baseline", default="stock")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    framework = DEFAULT_FRAMEWORK
    if argv[:1] == ["--framework"] and len(argv) >= 2:
        framework, argv = argv[1], argv[2:]
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        print("stages:", ", ".join(STAGE_NAMES))
        return 0
    command = argv[0].replace("-", "_")
    adapter = resolve_adapter(framework)
    if command in GROUPS:
        args = composite_parser(command).parse_args(argv[1:])
        ctx = E2EContext.create(framework, args.model, args.isa, args.run_dir, args.force)
        options = Options(
            gguf_head=getattr(args, "gguf_head", None), label=getattr(args, "label", None),
            agent_build=getattr(args, "agent_build", False),
            kernel_set_args=shlex.split(getattr(args, "kernel_set_args", "")),
            baseline=getattr(args, "baseline", "stock"),
        )
        return run_group(command, ctx, adapter, options, args.start, args.stop)
    if command in STAGE_NAMES:
        return adapter.entrypoint(command)(argv[1:])
    print(f"e2e: unknown command {argv[0]!r}\n", file=sys.stderr)
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
