"""
eval/run_benchmark.py — Agentic benchmark CLI for arm-bench.

Full end-to-end: provision (if needed) → run agentic LLM eval → score → (optionally) teardown.

Usage:
    # Single definition by name or op_type prefix:
    python eval/run_benchmark.py --problem conv2d --dataset ncnn --model anthropic/claude-opus-4-8

    # All definitions for a dataset:
    python eval/run_benchmark.py --all --dataset ncnn --model anthropic/claude-opus-4-8

    # Provision a fresh instance, run, then tear it down:
    python eval/run_benchmark.py --all --dataset ncnn --model anthropic/claude-opus-4-8 \\
        --provision --teardown

    # Override ISA (e.g. Graviton3 SVE):
    python eval/run_benchmark.py --all --dataset simd-loop --model anthropic/claude-opus-4-8 \\
        --isa sve
"""

import argparse
import base64
import json
import time
from pathlib import Path

from dotenv import load_dotenv

from bench.config import BenchmarkConfig
from bench.data.trace_set import TraceSet
from eval.config import REPO_ROOT
from eval.evaluator import run_agentic_eval
from eval.provision import (
    get_or_provision,
    get_running_instance,
    teardown,
    provision,
    ISA_INSTANCE_MAP,
    InstanceHandle,
)

BENCH_TRACE = REPO_ROOT / "bench-trace"
RESULTS_DIR = REPO_ROOT / "results"
load_dotenv(REPO_ROOT / ".env")

# Dataset → bench.cli collect-baselines --baseline-author value.
# Must match the baseline_author used by AgentTools (BenchmarkConfig default in
# eval/agent_tools/base.py is "reference-scalar"), so speedup computation works.
_DATASET_BASELINE_AUTHOR: dict[str, str] = {
    "ncnn": "baseline-ncnn-arm",
    "simd-loop": "reference",
    "llama.cpp": "baseline-llamacpp-arm",
}


def _author_from_model(model: str) -> str:
    """Derive a short author label from the model string."""
    return model.split("/")[-1]


def _defs_for_dataset(ts: TraceSet, dataset: str) -> list:
    """Return definitions that have at least one solution in the given dataset.

    Uses the solution metadata already loaded by TraceSet — no filesystem scan.
    """
    matching = {
        def_name
        for def_name, sols in ts.solutions.items()
        if any(s.dataset.value == dataset for s in sols)
    }
    return [ts.definitions[n] for n in sorted(matching) if n in ts.definitions]


def _ensure_baselines(
    handle: InstanceHandle,
    definitions: list,
    baseline_author: str,
    verbose: bool = True,
) -> None:
    """Lazily collect baseline traces on the remote for any definitions missing them.

    Runs one SSH call to find definitions that lack traces for `baseline_author`,
    then calls `bench.cli collect-baselines` per missing definition (idempotent).
    The check is author-specific — a trace file for a *different* author does not
    satisfy the requirement.
    """
    if not definitions:
        return

    # Build a Python script that checks each definition for author-specific traces,
    # then encode it as base64 to avoid all shell-quoting issues.
    check_code = (
        "import json; from pathlib import Path\n"
        "bench = Path.home() / 'arm-bench' / 'bench-trace'\n"
        f"auth = {baseline_author!r}\n"
        f"names = {[d.name for d in definitions]!r}\n"
        "td = bench / 'traces'\n"
        "for n in names:\n"
        "    found = False\n"
        "    if td.exists():\n"
        "        for f in sorted(td.rglob(n + '.jsonl')):\n"
        "            try:\n"
        "                for line in f.open():\n"
        "                    line = line.strip()\n"
        "                    if line and json.loads(line).get('solution', '').startswith(auth):\n"
        "                        found = True; break\n"
        "            except Exception:\n"
        "                pass\n"
        "            if found: break\n"
        "    if not found: print(n)\n"
    )
    b64 = base64.b64encode(check_code.encode()).decode()
    _, out, _ = handle.run(
        f"echo {b64!r} | base64 -d | python3",
        timeout=30,
    )
    missing_names = set(out.split())
    missing = [d for d in definitions if d.name in missing_names]

    if not missing:
        if verbose:
            print(f"[baselines] All {len(definitions)} baseline trace(s) present.")
        return

    if verbose:
        print(
            f"\n[baselines] {len(missing)}/{len(definitions)} definition(s) missing baseline "
            f"traces — collecting (author={baseline_author!r})..."
        )

    for i, d in enumerate(missing):
        if verbose:
            print(f"  [{i+1}/{len(missing)}] {d.name} ...", end=" ", flush=True)
        rc, out, err = handle.run(
            f"cd ~/arm-bench && python3 -m bench.cli collect-baselines "
            f"--baseline-author {baseline_author} --definition {d.name}",
            timeout=600,
        )
        if verbose:
            if rc == 0:
                print("OK")
            else:
                combined = "\n".join(filter(None, [out.strip(), err.strip()]))
                print(f"WARNING: {combined}")


def main():
    parser = argparse.ArgumentParser(
        description="Agentic LLM benchmark for arm-bench (ncnn / simd-loop)"
    )

    # Dataset selection
    parser.add_argument(
        "--dataset", default="ncnn", choices=["ncnn", "simd-loop", "llama.cpp"],
        help="Dataset to benchmark (default: ncnn)",
    )

    # Problem selection
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--problem",
        help=(
            "Definition name or op_type prefix to run "
            "(e.g. 'conv2d_kh3_kw3_sh1_sw1' or just 'conv2d')"
        ),
    )
    grp.add_argument("--all", action="store_true", help="Run all definitions for the dataset")

    # Model
    parser.add_argument("--model", required=True,
                        help="LiteLLM model string, e.g. anthropic/claude-opus-4-8")

    # Instance lifecycle
    parser.add_argument("--provision", action="store_true",
                        help="Provision a new instance even if one is already configured")
    parser.add_argument("--teardown", action="store_true",
                        help="Destroy the instance after evaluation")
    parser.add_argument(
        "--isa", default=None, choices=["neon", "sve", "sve2", "sme2"],
        help="ISA target (default: sve2). Determines the EC2 instance type via ISA_INSTANCE_MAP.",
    )

    # Eval options
    parser.add_argument("--max-turns", type=int, default=20,
                        help="Max agent turns per definition (default: 20)")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-turn output")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to results/")
    parser.add_argument("--save-trace", action="store_true",
                        help="Save full version_history to traces/ alongside results")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Skip lazy baseline collection (use if baselines are already present)")

    args = parser.parse_args()

    # ── Resolve ISA → instance type ────────────────────────────────────────
    isa = args.isa or "sve2"
    instance_type = ISA_INSTANCE_MAP.get(isa, "c8g.large")

    if args.provision:
        handle = provision(instance_type, dataset=args.dataset)
    else:
        handle = get_running_instance(isa)
        if handle is None:
            print(f"No running {instance_type} instance ({isa}). Provisioning...")
            handle = provision(instance_type, dataset=args.dataset)

    # ── Load TraceSet + filter definitions ────────────────────────────────
    ts = TraceSet.from_path(BENCH_TRACE)

    if args.problem:
        # Match by op_type within the selected dataset
        dataset_defs = _defs_for_dataset(ts, args.dataset)
        problem_defs = [d for d in dataset_defs if d.op_type == args.problem]
        if not problem_defs:
            available = sorted({d.op_type for d in dataset_defs})
            print(f"No definitions matching {args.problem!r} in dataset {args.dataset!r}.")
            print(f"Available op_types: {available}")
            return
    else:
        # --all: restrict to definitions that belong to the selected dataset
        problem_defs = _defs_for_dataset(ts, args.dataset)

    print(f"Running {len(problem_defs)} definition(s) "
          f"(dataset: {args.dataset}, model: {args.model}, instance: {instance_type})")

    # ── Lazy baseline collection ──────────────────────────────────────────
    baseline_author = _DATASET_BASELINE_AUTHOR.get(args.dataset, "reference-scalar")
    if not args.skip_baselines:
        _ensure_baselines(handle, problem_defs, baseline_author, verbose=not args.quiet)

    # bench_cfg carries the correct baseline_author for this dataset so that
    # AgentTools (and DefaultEvaluator on the remote) use the same author when
    # computing speedup — otherwise speedup is always None.
    bench_cfg = BenchmarkConfig(baseline_author=baseline_author)

    # ── Run evaluations ───────────────────────────────────────────────────
    author = _author_from_model(args.model)
    results: dict[str, dict] = {}
    RESULTS_DIR.mkdir(exist_ok=True)

    for i, defn in enumerate(problem_defs):
        print(f"\n[{i+1}/{len(problem_defs)}] {defn.name}")
        try:
            result = run_agentic_eval(
                definition=defn,
                trace_set=ts,
                author=author,
                model=args.model,
                handle=handle,
                dataset=args.dataset,
                bench_cfg=bench_cfg,
                max_turns=args.max_turns,
                verbose=not args.quiet,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            result = {
                "status": "ERROR",
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "version_history": [],
            }

        results[defn.name] = result

        if not args.no_save:
            stem = f"{defn.name}_{args.dataset}_{args.model.replace('/', '_')}"
            data = {
                **{k: v for k, v in result.items() if k != "version_history"},
                "definition": defn.name,
                "dataset": args.dataset,
                "model": args.model,
            }

            out = RESULTS_DIR / f"{stem}.json"
            out.write_text(json.dumps(data, indent=2))

            jsonl_out = RESULTS_DIR / f"{stem}.jsonl"
            with jsonl_out.open("a") as f:
                f.write(json.dumps(data) + "\n")

            if args.save_trace and result.get("version_history"):
                traces_dir = REPO_ROOT / "traces"
                traces_dir.mkdir(exist_ok=True)
                ts_stamp = result.get("timestamp", "").replace(":", "-")
                trace_out = traces_dir / f"{stem}_{ts_stamp}.json"
                trace_out.write_text(json.dumps({
                    "definition": defn.name,
                    "dataset": args.dataset,
                    "model": args.model,
                    "timestamp": result.get("timestamp"),
                    "version_history": result.get("version_history"),
                }, indent=2))

    # ── Print summary ─────────────────────────────────────────────────────
    _print_summary(results, args.dataset, args.model)

    # ── Teardown ──────────────────────────────────────────────────────────
    if args.teardown:
        print("\n[teardown] Destroying instance...")
        teardown()
    else:
        handle = get_running_instance(instance_type)
        if handle and handle.host:
            print(
                f"\n[WARNING] Instance at {handle.host} is still running and accruing cost. "
                f"Run with --teardown to destroy it after evaluation, "
                f"or: python -m eval.provision --teardown"
            )


def _print_summary(results: dict[str, dict], dataset: str, model: str):
    n = len(results)
    if n == 0:
        return

    n_passed = sum(1 for r in results.values() if r.get("status") == "PASSED")
    time_speedups = [
        r["time_speedup"]
        for r in results.values()
        if r.get("time_speedup") is not None
    ]
    cycle_speedups = [
        r["cycle_speedup"]
        for r in results.values()
        if r.get("cycle_speedup") is not None
    ]

    def _geomean(vals: list[float]) -> float | None:
        if not vals:
            return None
        product = 1.0
        for v in vals:
            product *= v
        return product ** (1.0 / len(vals))

    gm_time = _geomean(time_speedups)
    gm_cycle = _geomean(cycle_speedups)

    print(f"\n{'='*60}")
    print(f"  Benchmark Summary")
    print(f"  Dataset: {dataset}  |  Model: {model}")
    print(f"{'='*60}")
    print(f"  Total definitions:        {n}")
    print(f"  Passed:                   {n_passed}/{n}  ({100*n_passed//n}%)")
    if gm_time is not None:
        print(f"  Geomean time speedup:     {gm_time:.3f}×")
    if gm_cycle is not None:
        print(f"  Geomean cycle speedup:    {gm_cycle:.3f}×")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
