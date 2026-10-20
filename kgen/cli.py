"""One entry point for generating kernel material into bench-trace.

  python -m kgen add definition <file> | workload <file> --definition <name> | solution <file> [--replace]
  python -m kgen check <definition> [--static-only] [--num-selected-workloads N] [--isa sve2]
  python -m kgen show <definition> [--full]
  python -m kgen discover llamacpp <gguf head file> [--out inventory.json]
  python -m kgen capture pytorch --loader load.py --source <text> (--definitions N... | --tag T)
  python -m kgen calibrate --author A --baseline-author B (--definitions N... | --tag T)

`add` lints a file the agent wrote and places it under --root (default bench-trace) at the conventional path;
it refuses when the target exists unless --replace, and writes nothing when the file has problems.
`check` and `show` read the warehouse; `discover` finds a framework's operators from its model, `capture` gives
workloads real inputs from a model run and `calibrate` records what the reference scores on them. Bulk generators
for one family of kernels live in scripts/, not here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .add import AddError, add_definition, add_solution, add_workload
from .calibrate import calibrate_sqnr_floors
from .capture import capture_pytorch
from .check import check
from .discover.llamacpp import build_inventory
from .errors import KgenError
from .show import show

REPO = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m kgen", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add", help="lint a file and place it in bench-trace")
    add.add_argument("--root", type=Path, default=REPO / "bench-trace", help="the warehouse to write into")
    add.add_argument("--replace", action="store_true", help="overwrite the target if it exists")
    kinds = add.add_subparsers(dest="kind", required=True)
    kinds.add_parser("definition", help="a Definition JSON").add_argument("file", type=Path)
    workload = kinds.add_parser("workload", help="a JSONL of workloads (uuid optional: kept if given, else the existing one for the same axes, else new)")
    workload.add_argument("file", type=Path)
    workload.add_argument("--definition", required=True, help="name of the definition they belong to")
    kinds.add_parser("solution", help="a Solution JSON").add_argument("file", type=Path)
    shw = commands.add_parser("show", help="what bench-trace holds for a definition")
    shw.add_argument("definition")
    shw.add_argument("--root", type=Path, default=REPO / "bench-trace")
    shw.add_argument("--full", action="store_true", help="also print the reference")
    dis = commands.add_parser("discover", help="the operators of a framework's model (frameworks only)")
    frameworks = dis.add_subparsers(dest="framework", required=True)
    llamacpp = frameworks.add_parser("llamacpp", help="matmul shapes of a model from its GGUF header")
    llamacpp.add_argument("gguf", type=Path, help="the GGUF file, or just its first part")
    llamacpp.add_argument("--out", type=Path, help="write the inventory JSON here")
    cap = commands.add_parser("capture", help="real activations into the workloads")
    sources = cap.add_subparsers(dest="source_kind", required=True)
    torch_source = sources.add_parser("pytorch", help="run a PyTorch model and capture the inputs of its Linear / Conv2d layers")
    torch_source.add_argument("--loader", type=Path, required=True, help="a python file whose load() returns (model, batches)")
    torch_source.add_argument("--source", required=True, help="provenance text recorded on each input")
    torch_source.add_argument("--definitions", nargs="*", default=[], help="definitions to give real inputs")
    torch_source.add_argument("--tag", help="also every definition carrying this tag, e.g. e2e:qwen3.5-4b")
    torch_source.add_argument("--input", dest="input_name", help="which input of the definitions to replace (default: the first)")
    torch_source.add_argument("--modules-per-shape", type=int, default=4, help="layers of the same shape to sample, spread over the model")
    torch_source.add_argument("--calls-per-module", type=int, default=2, help="forward calls recorded per sampled layer")
    torch_source.add_argument("--max-rows", type=int, default=4096, help="rows kept from one call of a Linear layer")
    torch_source.add_argument("--root", type=Path, default=REPO / "bench-trace")
    torch_source.add_argument("--dry-run", action="store_true")
    cal = commands.add_parser("calibrate", help="record the baseline's SQNR on each workload (runs kernels: aarch64)")
    cal.add_argument("--author", required=True, help="whose solution to measure (the reference implementation)")
    cal.add_argument("--baseline-author", required=True, help="the solution the evaluator scores speedup against")
    cal.add_argument("--definitions", nargs="*", default=[])
    cal.add_argument("--tag", help="also every definition carrying this tag, e.g. e2e:qwen3.5-4b "
                                   "(with neither --definitions nor --tag: every correctness:sqnr definition)")
    cal.add_argument("--root", type=Path, default=REPO / "bench-trace")
    cal.add_argument("--dry-run", action="store_true")
    chk = commands.add_parser("check", help="check a definition, its workloads and solutions")
    chk.add_argument("definition")
    chk.add_argument("--root", type=Path, default=REPO / "bench-trace", help="the warehouse to read")
    chk.add_argument("--static-only", action="store_true", help="lint only: no execution")
    chk.add_argument("--num-selected-workloads", type=int, metavar="N",
                     help="run the execution legs on N randomly chosen workloads (default: all)")
    chk.add_argument("--seed", type=int, default=0, help="seed of that choice, so a rerun checks the same workloads")
    chk.add_argument("--isa", help="picks the baseline author where a dataset has one per isa (simd-loop)")
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
    if args.command == "show":
        print(show(args.root, args.definition, args.full))
        return 0
    if args.command == "discover":
        build_inventory(args.gguf, args.out)
        return 0
    if args.command in ("capture", "calibrate"):
        try:
            if args.command == "capture":
                capture_pytorch(args.root, args.loader, args.source, args.definitions, args.tag, args.input_name,
                                args.modules_per_shape, args.calls_per_module, args.max_rows, args.dry_run)
            else:
                _, problems = calibrate_sqnr_floors(args.root, author=args.author, baseline_author=args.baseline_author,
                                                    definitions=args.definitions, tag=args.tag, dry_run=args.dry_run)
                return 1 if problems else 0
        except KgenError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        return 0
    if args.command == "check":
        legs, status = check(args.root, args.definition, args.static_only, args.isa,
                             args.num_selected_workloads, args.seed)
        for leg in legs:
            print(f"[{leg.status.upper():>4}] {leg.name}")
            for line in leg.lines:
                print(f"       {line}")
        return status
    return 0


if __name__ == "__main__":
    sys.exit(main())
