"""armbench CLI. Thin argparse layer over Benchmark / TraceSet.

Usage:
    armbench bench    --root <path> --definition <name> --solution <name> (--dataset <name> | --baseline-author <author>)
                      [--workload-axes axis=value,...] [--isa <isa>]
    armbench collect-baselines --root <path> (--dataset <name> | --baseline-author <author>)
                      [--definition <name>] [--isa <isa>]
    armbench bench-author --root <path> --dataset <name> [--author <name>] [--op-type/--definition/--isa ...]
    armbench retime   --root <path> --dataset <name> --isa <isa> --index <extracted.json> --pass-id <id> --out <file>
    armbench summary  --root <path>
    armbench list-definitions --root <path>
    armbench list-solutions   --root <path> [--definition <name>]

"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

from bench import retime
from bench.benchmark import Benchmark, BenchmarkConfig
from bench.data import TraceSet
from contracts import BASELINE_AUTHORS, BASELINE_AUTHORS_BY_ISA, ISA_TABLE, REFERENCE_SCALAR_AUTHORS, baseline_author_for


def _default_root() -> Path:
    """Default warehouse root: arm-bench/bench-trace/.

    The TraceSet expects to find definitions/ + solutions/ + workloads/ +
    traces/ directly under this root.
    """
    return Path(__file__).resolve().parent.parent / "bench-trace"


def _parse_axes_filter(s: Optional[str]) -> Optional[Dict[str, int]]:
    """Parse 'N=1,H=56,W=56' → {'N': 1, 'H': 56, 'W': 56}."""
    if not s:
        return None
    out: Dict[str, int] = {}
    for pair in s.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(f"--workload-axes pair must be 'name=value': {pair!r}")
        k, v = pair.split("=", 1)
        out[k.strip()] = int(v.strip())
    return out


def require_isa_for(datasets, isa: Optional[str]) -> None:
    """A dataset with a different baseline per isa must not silently fall back to its default."""
    needing = sorted(set(datasets) & set(BASELINE_AUTHORS_BY_ISA))
    if needing and not isa:
        raise SystemExit(f"--isa is required for dataset {', '.join(needing)}: its baseline differs per isa "
                         f"(choose from {sorted(ISA_TABLE)}), or pass --baseline-author")


def baseline_author_of(explicit: Optional[str], isa: Optional[str], dataset: Optional[str]) -> str:
    """The author whose solutions are the speedup baseline of a run: `--baseline-author` if given, else the
    baseline of the explicitly passed `--dataset` on `--isa`. The dataset is never inferred.
    """
    if explicit:
        return explicit
    if dataset is None:
        raise SystemExit("cannot tell the baseline of this run: pass --baseline-author, or --dataset")
    if dataset not in BASELINE_AUTHORS:
        raise SystemExit(f"dataset {dataset!r} has no baseline author in config/kernel_contracts.yaml; "
                         f"pass --baseline-author (datasets with one: {sorted(BASELINE_AUTHORS)})")
    require_isa_for([dataset], isa)
    author = baseline_author_for(dataset, isa)
    print(f"baseline author: {author}  (dataset {dataset}" + (f", isa {isa}" if isa else "") + ")")
    return author


# ── Subcommand dispatchers ───────────────────────────────────────────────────

def cmd_bench(args: argparse.Namespace) -> int:
    ts = TraceSet.from_path(args.root)
    baseline_author = baseline_author_of(args.baseline_author, args.isa, args.dataset)
    bench = Benchmark(ts, BenchmarkConfig(baseline_author=baseline_author))
    try:
        traces = bench.bench(
            definition=args.definition,
            solution=args.solution,
            workload_filter=_parse_axes_filter(args.workload_axes),
        )
    finally:
        bench.close()
    for t in traces:
        ev = t.evaluation
        if ev is None:
            print(f"  {t.workload.uuid[:8]}  (no evaluation)")
            continue
        if ev.status.value == "PASSED":
            perf = ev.performance
            corr = ev.correctness
            speedup = f"  speedup={perf.time_speedup:.3f}x" if perf.time_speedup is not None else ""
            print(
                f"  {t.workload.uuid[:8]}  PASSED  "
                f"min={perf.min_ns / 1000:.2f}us  p5={perf.p5_ns / 1000:.2f}us  "
                f"max_abs={corr.max_absolute_error:.2e}  max_rel={corr.max_relative_error:.2e}{speedup}"
            )
        else:
            print(f"  {t.workload.uuid[:8]}  {ev.status.value}  {ev.log[:120]}")
    return 0


def cmd_bench_author(args: argparse.Namespace) -> int:
    """Run every solution of one author in one dataset and report which pass: the smoke test for a reference set."""
    if not args.dataset:
        raise SystemExit("pass --dataset (never inferred from the author)")
    author = args.author or REFERENCE_SCALAR_AUTHORS.get(args.dataset)
    if not author:
        raise SystemExit(f"dataset {args.dataset!r} has no reference-scalar author: pass --author")
    require_isa_for([args.dataset], args.isa)
    ts = TraceSet.from_path(args.root)
    bench = Benchmark(ts)       # every solution gets its own dataset's baseline author, see run_author
    try:
        results = bench.run_author(
            author, dataset=args.dataset, op_type=args.op_type, definition=args.definition,
            workload_filter=_parse_axes_filter(args.workload_axes), isa=args.isa, dump_traces=args.dump_traces,
        )
    finally:
        bench.close()
    if not results:
        print(f"No solutions of author {author!r} under {args.root}"
              + (f" for dataset {args.dataset!r}" if args.dataset else "") + ".")
        return 1
    failed, fully_passed, total_wl, passed_wl = [], 0, 0, 0
    for r in results:
        name = f"{r.solution.dataset.value:<10} {r.solution.name:<60}"
        if r.skipped or r.error:
            print(f"  [{'SKIP' if r.skipped else 'FAIL'}] {name} {r.skipped or r.error}")
            if r.error:
                failed.append(r.solution.name)
            continue
        n, p = len(r.traces), sum(1 for t in r.traces if t.is_successful())
        total_wl, passed_wl = total_wl + n, passed_wl + p
        print(f"  [{'OK  ' if p == n else 'FAIL'}] {name} {p}/{n}")
        if p == n:
            fully_passed += 1
        else:
            failed.append(r.solution.name)
            first = next((t for t in r.traces if not t.is_successful() and t.evaluation is not None), None)
            if first is not None:
                print(f"         -> {first.workload.uuid[:8]} {first.evaluation.status.value}: {first.evaluation.log[:160]}")
    ran = sum(1 for r in results if not r.skipped)
    print(f"\n{fully_passed}/{ran} solutions fully passed; {passed_wl}/{total_wl} workloads PASSED.")
    if failed:
        print("FAILED:", " ".join(failed))
    return 1 if failed else 0


def cmd_retime(args: argparse.Namespace) -> int:
    """Re-time the solutions an index lists (and optionally other authors') against baselines collected
    fresh in an isolated copy of the root; see bench/retime.py."""
    root = retime.isolated_root(args.root)
    print(f"isolated root: {root}")
    ts = TraceSet.from_path(root)
    if args.index:
        defs, jobs = retime.jobs_from_index(json.loads(Path(args.index).read_text()), args.extra_authors)
    else:
        defs, jobs = retime.jobs_from_authors(ts, args.author, args.dataset, args.extra_authors)
        if not defs:
            raise SystemExit(f"no solutions of {args.author} for dataset {args.dataset!r} under {args.root}")
    baseline_author = baseline_author_of(args.baseline_author, args.isa, args.dataset)
    bench = Benchmark(ts, BenchmarkConfig(baseline_author=baseline_author, definitions=defs))
    try:
        retime.retime(ts, bench, baseline_author, defs, jobs, args.pass_id, Path(args.out),
                      dataset=args.dataset, isa=args.isa)
    finally:
        bench.close()
    return 0


def cmd_collect_baselines(args: argparse.Namespace) -> int:
    ts = TraceSet.from_path(args.root)
    baseline_author = baseline_author_of(args.baseline_author, args.isa, args.dataset)
    cfg = BenchmarkConfig(
        baseline_author=baseline_author,
        definitions=[args.definition] if args.definition else None,
    )
    bench = Benchmark(ts, cfg)
    try:
        traces = bench.collect_baselines()
    finally:
        bench.close()
    if not traces:
        print(f"No baseline traces produced. Check that solutions/<dataset>/{baseline_author}/"
              f" contains Solutions for the selected definitions and definitions/ + workloads/ are populated.")
        return 1
    n_pass = sum(1 for t in traces if t.is_successful())
    print(f"\n{len(traces)} traces produced ({n_pass} PASSED, {len(traces) - n_pass} other).")
    if n_pass == 0:
        # Producing traces isn't success on its own
        return 1
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    ts = TraceSet.from_path(args.root)
    print(json.dumps(ts.summary(), indent=2))
    return 0


def cmd_list_definitions(args: argparse.Namespace) -> int:
    ts = TraceSet.from_path(args.root)
    for name in sorted(ts.definitions):
        d = ts.definitions[name]
        n_sols = len(ts.solutions.get(name, []))
        n_wls = len(ts.workloads.get(name, []))
        print(f"  {name:<55}  op={d.op_type:<12}  solutions={n_sols}  workloads={n_wls}")
    return 0


def cmd_list_solutions(args: argparse.Namespace) -> int:
    ts = TraceSet.from_path(args.root)
    sols = []
    if args.definition:
        sols = ts.solutions.get(args.definition, [])
    else:
        for s_list in ts.solutions.values():
            sols.extend(s_list)
    for s in sorted(sols, key=lambda x: x.name):
        print(f"  {s.name:<60}  def={s.definition}  author={s.author}  dataset={s.dataset.value}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="armbench", description="arm-bench CPU kernel benchmark CLI")
    p.add_argument(
        "--root", type=Path, default=_default_root(),
        help="arm-bench root dir (default: inferred from bench/ location).",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("bench", help="Run a Solution against a Definition's workloads.")
    b.add_argument("--definition", required=True, help="Definition name.")
    b.add_argument("--solution", required=True, help="Solution name.")
    b.add_argument("--dataset", default=None, choices=sorted(BASELINE_AUTHORS),
                   help="Dataset of the solution, whose baseline is the speedup reference. "
                        "Required unless --baseline-author is given.")
    b.add_argument(
        "--workload-axes", default=None,
        help="Filter workloads, e.g. 'N=1,H=56,W=56'. Only matching workloads are run.",
    )
    b.add_argument(
        "--baseline-author", default=None,
        help="Author folder under solutions/<dataset>/ to use as the speedup baseline. Default: the "
             "baseline of the solution's dataset on --isa (contracts.baseline_author_for).",
    )
    b.add_argument(
        "--isa", default=None, choices=sorted(ISA_TABLE),
        help="Tier the run stands for; only matters for datasets whose baseline differs per isa (simd-loop).",
    )
    b.set_defaults(func=cmd_bench)

    ba = sub.add_parser(
        "bench-author",
        help="Run every solution of one author in one dataset on its definition's workloads and report "
             "pass/fail (reference smoke test).",
    )
    ba.add_argument("--author", default=None,
                    help="Solution author. Default: --dataset's reference-scalar author.")
    ba.add_argument("--dataset", required=True, choices=sorted(BASELINE_AUTHORS), help="Only this dataset's solutions.")
    ba.add_argument("--op-type", default=None, help="Only definitions of this op_type.")
    ba.add_argument("--definition", default=None, help="Only this definition.")
    ba.add_argument("--workload-axes", default=None, help="Filter workloads, e.g. 'M=1'.")
    ba.add_argument("--isa", default=None, choices=sorted(ISA_TABLE),
                    help="Tier the run stands for; picks the per-isa baseline where a dataset has one.")
    ba.add_argument("--dump-traces", action="store_true", help="Also persist the traces (default: report only).")
    ba.set_defaults(func=cmd_bench_author)

    rt = sub.add_parser(
        "retime",
        help="Re-time stored solutions against freshly collected baselines (isolated root) and write one "
             "JSON-lines record each: geomean speedup, per-workload min_ns, the speedup originally reported.",
    )
    rt.add_argument("--dataset", required=True, choices=sorted(BASELINE_AUTHORS))
    rt.add_argument("--isa", required=True, choices=sorted(ISA_TABLE), help="Tier of this machine.")
    which = rt.add_mutually_exclusive_group(required=True)
    which.add_argument("--index", type=Path,
                       help="extracted.json from scripts/retime/extract_run_solutions.py: the solutions it wrote.")
    which.add_argument("--author", nargs="+",
                       help="Re-time every solution of these authors (of --dataset) in the warehouse.")
    rt.add_argument("--extra-authors", nargs="*", default=[],
                    help="Also re-time these authors on the same definitions, as controls. 'autovec' is not "
                         "looked up: it is the dataset's scalar starting point compiled like an agent candidate.")
    rt.add_argument("--pass-id", required=True, help="Label stored with every record; a rerun skips its done jobs.")
    rt.add_argument("--out", required=True, type=Path, help="JSON-lines file the records are appended to.")
    rt.add_argument("--baseline-author", default=None,
                    help="Default: the baseline of --dataset on --isa (contracts.baseline_author_for).")
    rt.set_defaults(func=cmd_retime)

    cb = sub.add_parser(
        "collect-baselines",
        help="Run every baseline-author Solution against its Definition's workloads, "
             "caching the timings for later speedup lookup.",
    )
    cb.add_argument("--baseline-author", default=None,
                    help="Author folder under solutions/<dataset>/ to treat as baseline. Default: the "
                         "baseline of --dataset on --isa.")
    cb.add_argument("--dataset", default=None, choices=sorted(BASELINE_AUTHORS),
                    help="Dataset whose baseline to collect. Required unless --baseline-author is given.")
    cb.add_argument("--isa", default=None, choices=sorted(ISA_TABLE),
                    help="Tier the baselines are collected on; only matters for datasets with a per-isa baseline.")
    cb.add_argument("--definition", default=None,
                    help="If set, only collect baselines for this Definition.")
    cb.set_defaults(func=cmd_collect_baselines)

    s = sub.add_parser("summary", help="Dump warehouse summary as JSON.")
    s.set_defaults(func=cmd_summary)

    ld = sub.add_parser("list-definitions", help="List all definitions in the warehouse.")
    ld.set_defaults(func=cmd_list_definitions)

    ls = sub.add_parser("list-solutions", help="List all solutions.")
    ls.add_argument("--definition", default=None, help="Filter by definition name.")
    ls.set_defaults(func=cmd_list_solutions)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
