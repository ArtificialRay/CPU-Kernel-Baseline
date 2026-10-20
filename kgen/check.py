"""`kgen check`: does what the agent wrote hold together, and agree with the real library?

Four legs, each reported on its own:
  static          the definition, its workloads and its solutions pass the same lint `add` applies
  reference       the definition's Python reference runs on every workload and returns the declared shape
  reference-scalar the dataset's reference-scalar solution builds, runs and matches the reference
  baseline        the dataset's baseline solution (the library's own implementation) does the same

Reference and reference-scalar are both written by the agent, so agreeing with each other proves little; the
baseline calls the real library, so agreement with it is what shows the reference describes the real operator.
Both solutions are compared against the reference only (bench does not expose solution outputs):
  both pass            consistent
  both fail            the reference is the likeliest culprit
  only one fails       that solution is the likeliest culprit
This reports what it measured and that hint; it does not issue a verdict. Timing is not measured.
"""
from __future__ import annotations

import dataclasses
import json
import platform
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import contracts
from bench.config import BenchmarkConfig
from bench.data import Definition, Solution
from bench.data.trace import EvaluationStatus
from bench.data.workload import Workload
from bench.runner import run_solution_on_workloads
from bench.runtime.inputs import gen_inputs_for_workload, set_trace_root

from .axes import resolve_axes
from .layout import definition_dir
from .lint import lint_definition, lint_solution, lint_workloads

PASS, FAIL, SKIP = "pass", "fail", "skip"
MAX_DETAILS = 3          # failing workloads described per leg; the rest are only counted


@dataclass
class Leg:
    name: str
    status: str
    lines: list = field(default_factory=list)


def find_definition_file(root: Path, name: str) -> Path | None:
    return next((root / "definitions").glob(f"*/{name}.json"), None)


def solution_files(root: Path, definition: Definition) -> list:
    """Every solution file of the definition, whatever its dataset or author."""
    d = definition_dir(definition)
    return sorted([*root.glob(f"solutions/*/*/{d}/{definition.name}.json"),
                   *root.glob(f"solutions/simd-loop/*/{definition.name}/*.json")])


def load_workloads(root: Path, definition: Definition) -> tuple[list, list]:
    path = root / "workloads" / definition_dir(definition) / f"{definition.name}.jsonl"
    if not path.exists():
        return [], [f"{path} does not exist"]
    rows, problems = lint_workloads(path.read_text().splitlines(), definition, require_uuid=True, root=root)
    return [Workload.model_validate(r) for r in rows] if not problems else [], [f"{path}: {p}" for p in problems]


def static_leg(root: Path, name: str) -> tuple[Leg, Definition | None, list, list]:
    path = find_definition_file(root, name)
    if path is None:
        return Leg("static", FAIL, [f"definition {name!r} is not under {root / 'definitions'}"]), None, [], []
    definition, problems = lint_definition(json.loads(path.read_text()))
    workloads, wl_problems = ([], []) if definition is None else load_workloads(root, definition)
    solutions = []
    for file in solution_files(root, definition) if definition else []:
        solution, sol_problems = lint_solution(json.loads(file.read_text()), definition)
        problems += [f"{file}: {p}" for p in sol_problems]
        if solution is not None:
            solutions.append(solution)
    problems += wl_problems
    status = FAIL if problems else PASS
    summary = [] if problems else [f"{path.name}, {len(workloads)} workloads, {len(solutions)} solutions"]
    return Leg("static", status, problems or summary), definition, workloads, solutions


def reference_leg(definition: Definition, workloads: list, root: Path) -> Leg:
    set_trace_root(root)
    ns: dict = {}
    exec(definition.reference, ns)      # reference code is dataset content, as in bench.runner
    run = ns["run"]
    expected_names = list(definition.outputs)
    bad = []
    for w in workloads:
        try:
            inputs = gen_inputs_for_workload(definition, w)
            axes = resolve_axes(definition, w.axes, {n: inputs[n] for n, s in definition.inputs.items() if s.shape is None})
            out = run(**inputs)
            outs = out if isinstance(out, (tuple, list)) and len(expected_names) > 1 else [out]
            if len(outs) != len(expected_names):
                raise ValueError(f"returned {len(outs)} outputs, the definition declares {len(expected_names)}")
            for key, value in zip(expected_names, outs):
                got = tuple(np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value).shape)
                spec = definition.outputs[key].shape
                want = tuple(axes[a] for a in spec) if spec is not None else ()
                if got != want and not (got == (1,) and want == ()):
                    raise ValueError(f"output {key!r} has shape {got}, expected {want} from {spec} with {axes}")
        except Exception as e:
            bad.append(f"workload {w.axes}: {type(e).__name__}: {e}")
    if bad:
        return Leg("reference", FAIL, bad[:MAX_DETAILS] + ([f"... and {len(bad) - MAX_DETAILS} more"] if len(bad) > MAX_DETAILS else []))
    return Leg("reference", PASS, [f"ran on {len(workloads)} workloads, output shapes as declared"])


def solution_leg(name: str, definition: Definition, solution: Solution | None, workloads: list, baseline_author: str,
                 root: Path) -> Leg:
    if solution is None:
        return Leg(name, SKIP, [f"no solution for {definition.name}"])
    if platform.machine() != "aarch64":
        return Leg(name, SKIP, [f"{solution.name} needs an aarch64 machine to run; this one is {platform.machine()}"])
    set_trace_root(root)
    cfg = dataclasses.replace(BenchmarkConfig().resolve_eval_config(definition), warmup=0, repeat=1)
    is_baseline = solution.author == baseline_author or solution.dataset.value == "simd-loop"
    traces = run_solution_on_workloads(definition, solution, workloads, is_baseline=is_baseline, cfg=cfg)
    failed = [t for t in traces if t.evaluation.status != EvaluationStatus.PASSED]
    lines = [f"{solution.name}: {len(traces) - len(failed)}/{len(traces)} workloads passed "
             f"(abs_tol={cfg.abs_tol}, rel_tol={cfg.rel_tol}, matched>={cfg.required_matched_ratio}, "
             f"sqnr>={cfg.min_sqnr_db} dB where it applies)"]
    for t in failed[:MAX_DETAILS]:
        lines.append(f"workload {t.workload.axes}: {t.evaluation.status.value}: {(t.evaluation.log or '').strip()[:400]}")
    if len(failed) > MAX_DETAILS:
        lines.append(f"... and {len(failed) - MAX_DETAILS} more failing workloads")
    return Leg(name, FAIL if failed else PASS, lines)


def hint(scalar: Leg, baseline: Leg) -> str | None:
    if scalar.status == PASS and baseline.status == PASS:
        return "reference-scalar and baseline both match the reference: consistent"
    if scalar.status == FAIL and baseline.status == FAIL:
        return "both solutions disagree with the reference: the reference is the likeliest culprit"
    if scalar.status == FAIL and baseline.status == PASS:
        return "baseline matches the reference but reference-scalar does not: reference-scalar is the likeliest culprit"
    if scalar.status == PASS and baseline.status == FAIL:
        return "reference-scalar matches the reference but baseline does not: the baseline binding, or the library call, is the likeliest culprit"
    return None


def select_workloads(workloads: list, count: int | None, seed: int) -> list:
    """`count` workloads drawn at random (the same ones for the same seed), in file order; all when count is None."""
    if count is None or count >= len(workloads):
        return workloads
    chosen = set(random.Random(seed).sample(range(len(workloads)), count))
    return [w for i, w in enumerate(workloads) if i in chosen]


def check(root: Path, name: str, static_only: bool = False, isa: str | None = None,
          num_selected_workloads: int | None = None, seed: int = 0) -> tuple[list, int]:
    """Run the legs; returns them and an exit status: 0 all legs passed, 1 a leg failed, 2 no failure but a leg
    (the baseline in particular) was skipped. The execution legs (reference, reference-scalar, baseline) run on
    `num_selected_workloads` randomly chosen workloads, all of them by default; static always reads every file."""
    legs = []
    # phase 1: static check, files parse (`bench/data` models), names/paths/tags follow convention, workload axes exist in the
   # definition, solution entry_point matches its sources. 
   # `--static-only` stops here.
    static, definition, workloads, solutions = static_leg(Path(root), name)
    legs.append(static)
    if static.status == FAIL or static_only:
        return legs, 1 if static.status == FAIL else 0
    dataset = solutions[0].dataset.value if solutions else None
    authors = {s.author: s for s in solutions}
    reference_author = contracts.REFERENCE_SCALAR_AUTHORS.get(dataset)
    baseline_author = contracts.baseline_author_for(dataset, isa) if dataset in contracts.BASELINE_AUTHORS else None
    selected = select_workloads(workloads, num_selected_workloads, seed)
    if len(selected) < len(workloads):
        legs[0].lines.append(f"execution legs run on {len(selected)} of {len(workloads)} workloads (random, seed {seed})")
    # phase2: reference check: definition's python reference runs on every workload, output shapes/dtypes should match
    legs.append(reference_leg(definition, selected, Path(root)))
    # phase3: compiles, runs, matches the reference within the contract tolerance
    scalar = solution_leg("reference-scalar", definition, authors.get(reference_author), selected, baseline_author, Path(root))
    # phase4: same as reference-scalar solution
    baseline = solution_leg("baseline", definition, authors.get(baseline_author), selected, baseline_author, Path(root))
    if baseline.status == SKIP and authors.get(baseline_author) is None:
        baseline.lines.append("the reference has no real-library counterpart to be checked against")
    legs += [scalar, baseline]
    note = hint(scalar, baseline)
    if note:
        legs.append(Leg("diagnosis", PASS if "consistent" in note else FAIL, [note]))
    if any(l.status == FAIL for l in legs):
        return legs, 1
    return legs, 2 if any(l.status == SKIP for l in legs) else 0
