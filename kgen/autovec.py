"""`kgen autovec`: Check if a definition's baseline at least half as fast as what the compiler makes of the scalar starting point

A baseline that is more than `--max-slowdown` (default 2x) slower than the reference-scalar solution rebuilt with the
machine's own march flag (so the compiler may vectorize it) does not represent expert performance

For every selected definition that has both
  - the baseline `contracts.baseline_author_for(dataset, isa)` picks (or --baseline-author, to judge a candidate
    before it is registered), and
  - the scalar starting point `contracts.REFERENCE_SCALAR_AUTHORS[dataset]`,
the starting point is rebuilt with its own flags plus the tier's march, both are timed on the definition's workloads
under the same EvalConfig, and slowdown = geomean(baseline min_ns) / geomean(autovec min_ns). The worst single
workload is reported too, since a geomean can hide one bad point.

Run it on the tier's own machine (aarch64):

    python -m kgen autovec --definitions loop_001 loop_002 --isa sve
    python -m kgen autovec --tag e2e:qwen3.5-4b --isa sve2 --num-selected-workloads 5

Exit status 1 if any baseline is flagged or either side fails to build, run or pass correctness; 2 if nothing failed
but a definition was skipped.
"""
from __future__ import annotations

import math
import platform
from pathlib import Path

import contracts
from bench.compile.registry import BuilderRegistry
from bench.config import BenchmarkConfig
from bench.data import Solution
from bench.runner import run_solution_on_workloads
from bench.runtime.inputs import set_trace_root

from .capture import select_targets
from .check import load_workloads, select_workloads, solution_files
from .errors import KgenError

AUTOVEC_AUTHOR = "autovec-check"


def isa_satisfies(required: list[str], isa: str) -> bool:
    """True if `isa`'s feature set in config/kernel_contracts.yaml covers every token in `required` (typically a
    Solution's spec.isa_features). "portable" has the features of neon."""
    table = {**contracts.ISA_TABLE, "portable": contracts.ISA_TABLE["neon"]}
    if isa not in table:
        raise KgenError(f"unknown isa {isa!r}; supported: {sorted(table)}")
    return set(required) <= set(table[isa].features)


def autovec_build(start: Solution, march: str) -> Solution:
    """The scalar starting point, compiled with the tier's march flag added. Built through model_validate (not
    model_copy) so the solution hash, and with it the build cache key, is recomputed for the new flags."""
    data = start.model_dump(mode="json")
    data["name"] = f"{start.name}__{AUTOVEC_AUTHOR}"
    data["author"] = AUTOVEC_AUTHOR
    data["spec"]["compile_flags"] = list(start.spec.compile_flags) + march.split()
    return Solution.model_validate(data)


def min_ns_by_workload(traces: list) -> tuple[dict | None, str]:
    """{workload uuid: min_ns}, or (None, reason) if any workload failed."""
    failed = [t for t in traces if not t.is_successful()]
    if failed:
        ev = failed[0].evaluation
        status = ev.status.value if ev is not None else "NO_EVALUATION"
        log = (ev.log or "").strip().splitlines()[-1:] if ev is not None else []
        return None, f"{len(failed)}/{len(traces)} workloads {status}" + (f": {log[0][:100]}" if log else "")
    return {t.workload.uuid: max(t.evaluation.performance.min_ns, 1) for t in traces}, ""


def geomean(values) -> float:
    values = list(values)
    return math.exp(sum(math.log(v) for v in values) / len(values))


def autovec_check(root: Path, isa: str, definitions=None, tag=None, max_slowdown: float = 2.0,
                  baseline_author: str | None = None, num_selected_workloads: int | None = None,
                  seed: int = 0) -> int:
    """Compare the baseline of each selected definition with its autovectorized starting point. Returns the exit
    status: 0 all fine, 1 a baseline flagged or an error, 2 no failure but a definition skipped."""
    if platform.machine() != "aarch64":
        raise KgenError(f"autovec times kernels and needs the tier's own aarch64 machine; this one is {platform.machine()}")
    root = Path(root)
    march = contracts.ISA_TABLE[isa].march if isa in contracts.ISA_TABLE else None
    if march is None:
        raise KgenError(f"unknown isa {isa!r}; supported: {sorted(contracts.ISA_TABLE)}")
    selected = [d for d, _ in select_targets(root, definitions, tag)]
    set_trace_root(root)
    print(f"isa={isa} march={march!r}, flag if slowdown > {max_slowdown:g}x\n")

    flagged, errors, skipped, checked = [], 0, 0, 0
    try:
        for definition in selected:
            by_author = {}
            for file in solution_files(root, definition):
                solution = Solution.model_validate_json(file.read_text())
                by_author[solution.author] = solution
            dataset = next((s.dataset.value for s in by_author.values()), None)
            if dataset not in contracts.REFERENCE_SCALAR_AUTHORS:
                print(f"  [SKIP] {definition.name}: no solutions of a known dataset"); skipped += 1
                continue
            base_author = baseline_author or contracts.baseline_author_for(dataset, isa)
            start_author = contracts.REFERENCE_SCALAR_AUTHORS[dataset]
            baseline, start = by_author.get(base_author), by_author.get(start_author)
            if baseline is None or start is None:
                missing = " and ".join(a for a, s in ((base_author, baseline), (start_author, start)) if s is None)
                print(f"  [SKIP] {definition.name}: no solution by {missing}"); skipped += 1
                continue
            if not isa_satisfies(list(baseline.spec.isa_features), isa):
                print(f"  [SKIP] {definition.name}: baseline needs {baseline.spec.isa_features}, not in {isa}"); skipped += 1
                continue
            workloads, problems = load_workloads(root, definition)
            if problems or not workloads:
                print(f"  [ERROR] {definition.name}: " + ("; ".join(problems) or "no workloads")); errors += 1
                continue
            workloads = select_workloads(workloads, num_selected_workloads, seed)
            cfg = BenchmarkConfig(baseline_author=base_author).resolve_eval_config(definition)
            base_ns, base_err = min_ns_by_workload(run_solution_on_workloads(
                definition, baseline, workloads, is_baseline=True, cfg=cfg))
            auto_ns, auto_err = min_ns_by_workload(run_solution_on_workloads(
                definition, autovec_build(start, march), workloads, is_baseline=True, cfg=cfg))
            if base_ns is None or auto_ns is None:
                why = "; ".join(x for x in (base_err and f"baseline {base_err}", auto_err and f"autovec {auto_err}") if x)
                print(f"  [ERROR] {definition.name}: {why}"); errors += 1
                continue
            slowdown = geomean(base_ns.values()) / geomean(auto_ns.values())
            worst_uuid = max(base_ns, key=lambda u: base_ns[u] / auto_ns[u])
            worst = next(w for w in workloads if w.uuid == worst_uuid)
            over = slowdown > max_slowdown
            checked += 1
            if over:
                flagged.append(definition.name)
            print(f"  [{'FLAG' if over else 'OK'}] {definition.name:<48} baseline {geomean(base_ns.values()):12.0f} ns  "
                  f"autovec {geomean(auto_ns.values()):12.0f} ns  slowdown {slowdown:6.2f}x  "
                  f"worst {base_ns[worst_uuid] / auto_ns[worst_uuid]:.2f}x at {worst.axes}")
    finally:
        BuilderRegistry.get_instance().cleanup()

    print(f"\n{checked} checked, {len(flagged)} flagged, {errors} errors, {skipped} skipped")
    if flagged:
        print(f"flagged (baseline more than {max_slowdown:g}x slower than autovec): {' '.join(flagged)}")
    if flagged or errors:
        return 1
    return 2 if skipped else 0
