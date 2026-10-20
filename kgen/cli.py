"""One entry point for generating kernel material into bench-trace.

  python -m kgen add definition <file> | workload <file> --definition <name> | solution <file> [--replace]
  python -m kgen check <definition> [--static-only] [--num-selected-workloads N] [--isa sve2]
  python -m kgen list
  python -m kgen run <recipe> [--stages define workload solutions] [--root bench-trace] <recipe options>

`add` lints a file the agent wrote and places it under --root (default bench-trace) at the conventional path;
it refuses when the target exists unless --replace, and writes nothing when the file has problems.
`run` plans once, then runs the chosen stages in pipeline order (all three by default). Every
stage overwrites the files it writes. `python -m kgen run <recipe> --help` lists a recipe's options.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .add import AddError, add_definition, add_solution, add_workload
from .check import check
from .recipes import STAGES, registry

REPO = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m kgen", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list the recipes")
    add = commands.add_parser("add", help="lint a file and place it in bench-trace")
    add.add_argument("--root", type=Path, default=REPO / "bench-trace", help="the warehouse to write into")
    add.add_argument("--replace", action="store_true", help="overwrite the target if it exists")
    kinds = add.add_subparsers(dest="kind", required=True)
    kinds.add_parser("definition", help="a Definition JSON").add_argument("file", type=Path)
    workload = kinds.add_parser("workload", help="a JSONL of workloads (no uuid; kgen assigns them)")
    workload.add_argument("file", type=Path)
    workload.add_argument("--definition", required=True, help="name of the definition they belong to")
    kinds.add_parser("solution", help="a Solution JSON").add_argument("file", type=Path)
    chk = commands.add_parser("check", help="check a definition, its workloads and solutions")
    chk.add_argument("definition")
    chk.add_argument("--root", type=Path, default=REPO / "bench-trace", help="the warehouse to read")
    chk.add_argument("--static-only", action="store_true", help="lint only: no execution")
    chk.add_argument("--num-selected-workloads", type=int, metavar="N",
                     help="run the execution legs on N randomly chosen workloads (default: all)")
    chk.add_argument("--seed", type=int, default=0, help="seed of that choice, so a rerun checks the same workloads")
    chk.add_argument("--isa", help="picks the baseline author where a dataset has one per isa (simd-loop)")
    run = commands.add_parser("run", help="run stages of a recipe")
    recipes = run.add_subparsers(dest="recipe", required=True)
    for recipe in registry().values():
        p = recipes.add_parser(recipe.name, help=recipe.description)
        p.add_argument("--root", type=Path, default=REPO / "bench-trace", help="the warehouse to write into")
        p.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
        recipe.add_arguments(p)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "add":
        try:
            target = {"definition": lambda: add_definition(args.file, args.root, args.replace),
                      "workload": lambda: add_workload(args.file, args.definition, args.root, args.replace),
                      "solution": lambda: add_solution(args.file, args.root, args.replace)}[args.kind]()
        except AddError as e:
            print("\n".join(f"error: {p}" for p in e.problems), file=sys.stderr)
            return 1
        print(f"wrote {target}")
        return 0
    if args.command == "check":
        legs, status = check(args.root, args.definition, args.static_only, args.isa,
                             args.num_selected_workloads, args.seed)
        for leg in legs:
            print(f"[{leg.status.upper():>4}] {leg.name}")
            for line in leg.lines:
                print(f"       {line}")
        return status
    recipes = registry()
    if args.command == "list":
        for recipe in recipes.values():
            print(f"{recipe.name:<16} {recipe.description}")
        return 0
    recipe = recipes[args.recipe]
    plan = recipe.plan(args)
    for stage in STAGES:
        if stage in args.stages:
            print(f"[kgen] {stage}: {getattr(recipe, stage)(plan, args)} files written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
