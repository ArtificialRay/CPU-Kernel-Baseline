"""Re-time stored solutions against freshly collected baselines, under the current timing protocol.

A speedup is baseline time over candidate time, so it only means something next to a baseline timed
the same way, on the same machine, around the same moment. A pass therefore works in its own copy
of the dataset root (`isolated_root`: definitions, workloads and solutions linked, an empty
traces/), so its baselines are collected fresh and nothing is reused from an earlier timing
protocol or pass. Within a pass it collects the tier's baseline for the selected definitions, runs
every selected solution on all of its definition's workloads, and records geomean(time_speedup) --
the same number evaluate() reports -- plus each workload's min_ns. Records go to a JSON-lines file
as they come, and a rerun of the same pass id skips what the file already has.

    python -m bench.cli retime --dataset simd-loop --isa sve --index runs-sve/extracted.json \\
        --extra-authors reference autovec --pass-id a1 --out retime-a1.jsonl

The solutions come from an index (the extracted.json of scripts/retime/extract_run_solutions.py) or
straight from the warehouse, by author. `autovec` as an extra author is not looked up: it is the
dataset's scalar starting point compiled like an agent candidate (bench/autovec.py), for any dataset.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Set, Tuple

from bench.autovec import AUTOVEC_AUTHOR, autovec_build
from contracts import REFERENCE_SCALAR_AUTHORS

Job = Tuple[str, str, Optional[str], Optional[float]]
"""(author, definition, cell label, speedup the run originally reported)."""


def isolated_root(root: Path) -> Path:
    """A fresh root that shares `root`'s inputs but has its own (empty) traces."""
    tmp = Path(tempfile.mkdtemp(prefix="retime-root-"))
    for name in os.listdir(root):
        if name != "traces":
            (tmp / name).symlink_to((Path(root) / name).resolve())
    return tmp


def jobs_from_index(index: list, extra_authors: Sequence[str] = ()) -> Tuple[List[str], List[Job]]:
    """The definitions an index covers, and one job per index entry plus one per extra author
    and definition (their cell is the author's name and there is no reported speedup)."""
    defs = sorted({e["definition"] for e in index})
    jobs: List[Job] = [(e["author"], e["definition"], e.get("cell"), e.get("reported_speedup")) for e in index]
    jobs += [(a, d, a, None) for a in extra_authors for d in defs]
    return defs, jobs


def jobs_from_authors(ts, authors: Sequence[str], dataset: str,
                      extra_authors: Sequence[str] = ()) -> Tuple[List[str], List[Job]]:
    """Like jobs_from_index, but the solutions are `authors`' own in the warehouse (of `dataset`).

    The cell and the originally reported speedup are read from the solution's description when it is
    the JSON extract_run_solutions.py writes; otherwise the cell is the author and nothing was reported."""
    jobs: List[Job] = []
    for defn in sorted(ts.solutions):
        for s in ts.solutions[defn]:
            if s.author in authors and s.dataset.value == dataset:
                try:
                    note = json.loads(s.description or "")
                except ValueError:
                    note = {}
                note = note if isinstance(note, dict) else {}
                jobs.append((s.author, defn, note.get("cell") or s.author, note.get("reported_speedup")))
    jobs.sort(key=lambda j: (j[0], j[1]))
    defs = sorted({j[1] for j in jobs})
    jobs += [(a, d, a, None) for a in extra_authors for d in defs]
    return defs, jobs


def solution_for(ts, author: str, definition: str, dataset: str, isa: str):
    """The Solution a job runs: `author`'s own, or for `autovec` the dataset's scalar starting point built
    like an agent candidate for `isa`. None when there is none."""
    if author == AUTOVEC_AUTHOR:
        start = ts.get_baseline_solution(definition, REFERENCE_SCALAR_AUTHORS[dataset])
        return None if start is None else autovec_build(start, isa)
    return ts.get_baseline_solution(definition, author)


def speedup_geomean(traces) -> Optional[float]:
    """Geomean of the per-workload time_speedup; None unless every workload passed with a speedup."""
    ok = [t for t in traces if t.is_successful()]
    speedups = [t.evaluation.performance.time_speedup for t in ok]
    if ok and len(ok) == len(traces) and all(speedups):
        return math.exp(sum(math.log(s) for s in speedups) / len(speedups))
    return None


def finished_jobs(out: Path) -> Set[Tuple[str, str, str]]:
    """(pass id, author, definition) of every record already in `out`."""
    done: Set[Tuple[str, str, str]] = set()
    if Path(out).exists():
        for line in Path(out).read_text().splitlines():
            r = json.loads(line)
            done.add((r["pass_id"], r["author"], r["definition"]))
    return done


def retime(ts, bench, baseline_author: str, defs: Sequence[str], jobs: Sequence[Job], pass_id: str,
           out: Path, dataset: str, isa: str, log: Callable[..., None] = print) -> int:
    """Collect `baseline_author`'s baselines for `defs` into `ts`, then re-time every job and append
    one record per job to `out`. `ts` must be the isolated root's TraceSet and `bench` a Benchmark
    over it; `dataset` and `isa` say what an `autovec` job is built from and for. Returns how many
    records were written."""
    done = finished_jobs(out)
    t0 = time.time()
    log(f"[{pass_id}] collecting {baseline_author} for {len(defs)} definitions", flush=True)
    base = {}
    for t in bench.collect_baselines(dump_traces=True):
        perf = t.evaluation.performance if t.is_successful() else None
        base.setdefault(t.definition, {})[t.workload.uuid] = perf.min_ns if perf else None
    log(f"[{pass_id}] baselines done in {time.time() - t0:.0f}s; {len(jobs)} solutions", flush=True)

    written = 0
    with Path(out).open("a") as f:
        for i, (author, defn, cell, reported) in enumerate(jobs, 1):
            if (pass_id, author, defn) in done:
                continue
            sol = solution_for(ts, author, defn, dataset, isa)
            if sol is None:
                log(f"  missing solution {author} {defn}", flush=True)
                continue
            traces = bench.run_solution(ts.definitions[defn], sol, ts.get_workloads(defn), dump_traces=False)
            ok = [t for t in traces if t.is_successful()]
            gm = speedup_geomean(traces)
            bad = sorted({t.evaluation.status.value for t in traces if not t.is_successful()})
            rec = dict(pass_id=pass_id, author=author, cell=cell, definition=defn,
                       speedup=gm, reported=reported, status=bad or ["PASSED"],
                       workloads={t.workload.uuid: dict(
                           min_ns=t.evaluation.performance.min_ns,
                           base_min_ns=base.get(defn, {}).get(t.workload.uuid),
                           speedup=t.evaluation.performance.time_speedup) for t in ok})
            f.write(json.dumps(rec) + "\n")
            f.flush()
            written += 1
            log(f"  [{i}/{len(jobs)}] {cell or author:52} {defn:9} "
                f"{'%.3f' % gm if gm else '-':>7}  reported {'%.3f' % reported if reported else '-':>7}"
                f"  {'' if not bad else bad}", flush=True)
    log(f"[{pass_id}] done in {time.time() - t0:.0f}s", flush=True)
    return written
