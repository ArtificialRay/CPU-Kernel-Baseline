"""
eval/run_benchmark.py — Agentic benchmark CLI for arm-bench.

Full end-to-end: provision (if needed) → run agentic LLM eval → score → (optionally) teardown.
Supports multiple EC2 instances: definitions are distributed across instances in parallel
via ThreadPoolExecutor; each thread gets a sticky handle via BenchPipeline.

Usage:
    # Single definition by name or op_type prefix:
    python eval/run_benchmark.py --problem conv2d --dataset ncnn --model anthropic/claude-opus-4-8

    # All definitions for a dataset:
    python eval/run_benchmark.py --all --dataset ncnn --model anthropic/claude-opus-4-8

    # Multi-instance: run 2 definitions in parallel across 2 instances:
    python eval/run_benchmark.py --all --dataset ncnn --model anthropic/claude-opus-4-8 \\
        --num-instances 2

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

from bench.config import BenchmarkConfig
from bench.data.trace_set import TraceSet
from eval.bench_pipeline import BenchPipeline
from eval.config import REPO_ROOT
from eval.evaluator import run_agentic_eval
from eval.provision import (
    get_running_instances,
    teardown,
    provision_n,
    ISA_INSTANCE_MAP,
    InstanceHandle,
)

load_dotenv(REPO_ROOT / ".env")

BENCH_TRACE = REPO_ROOT / "bench-trace"

# Dataset → bench.cli collect-baselines --baseline-author value.
# Must match the baseline_author used by AgentTools (BenchmarkConfig default in
# eval/agent_tools/base.py is "reference-scalar"), so speedup computation works.
_DATASET_BASELINE_AUTHOR: dict[str, str] = {
    "ncnn": "baseline-ncnn-arm",
    "simd-loop": "reference-scalar",
}


class BaselineCollectionError(RuntimeError):
    """Raised when baseline collection fails for a definition; aborts the entire run."""

    def __init__(self, def_name: str, host: str, detail: str) -> None:
        super().__init__(
            f"Baseline collection failed for {def_name!r} on {host}: {detail}"
        )
        self.def_name = def_name
        self.host = host


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

    Raises BaselineCollectionError immediately on any failure — no baseline means
    no speedup metric, making the agent run meaningless.
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
            print(f"[baselines] All {len(definitions)} baseline trace(s) present on {handle.host}.")
        return

    if verbose:
        print(
            f"\n[baselines] {len(missing)}/{len(definitions)} definition(s) missing baseline "
            f"traces on {handle.host} — collecting (author={baseline_author!r})..."
        )

    for i, d in enumerate(missing):
        if verbose:
            print(f"  [{i+1}/{len(missing)}] {d.name} ...", end=" ", flush=True)
        rc, out, err = handle.run(
            f"cd ~/arm-bench && python3 -m bench.cli collect-baselines "
            f"--baseline-author {baseline_author} --definition {d.name}",
            timeout=600,
        )
        if rc == 0:
            if verbose:
                print("OK")
        else:
            combined = "\n".join(filter(None, [out.strip(), err.strip()]))
            raise BaselineCollectionError(d.name, handle.host, combined)


def _ensure_baselines_parallel(
    handles: list[InstanceHandle],
    definitions: list,
    baseline_author: str,
    verbose: bool = True,
) -> None:
    """Run _ensure_baselines across all handles in parallel.

    Distributes definitions round-robin across handles. Any BaselineCollectionError
    from any handle is re-raised immediately (first error wins).
    """
    if not definitions or not handles:
        return

    # Distribute definitions across handles (round-robin chunks)
    chunks = [definitions[i::len(handles)] for i in range(len(handles))]

    errors: list[BaselineCollectionError] = []
    with ThreadPoolExecutor(max_workers=len(handles)) as pool:
        futures = {
            pool.submit(_ensure_baselines, handle, chunk, baseline_author, verbose): handle
            for handle, chunk in zip(handles, chunks)
            if chunk
        }
        for fut in as_completed(futures):
            try:
                fut.result()
            except BaselineCollectionError as e:
                errors.append(e)

    if errors:
        raise errors[0]


def _save_result(defn, result: dict, args) -> None:
    """Write result summary to agent-runs/<def>/result.json."""
    if args.no_save:
        return

    run_dir = REPO_ROOT / "agent-runs" / defn.name
    run_dir.mkdir(parents=True, exist_ok=True)

    include_history = args.save_trace and bool(result.get("version_history"))
    data = {
        **{k: v for k, v in result.items() if k != "version_history" or include_history},
        "definition": defn.name,
        "dataset": args.dataset,
        "model": args.model,
    }
    (run_dir / "result.json").write_text(json.dumps(data, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Agentic LLM benchmark for arm-bench (ncnn / simd-loop)"
    )

    # Dataset selection
    parser.add_argument(
        "--dataset", default="ncnn", choices=["ncnn", "simd-loop"],
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
    parser.add_argument(
        "--num-instances", type=int, default=1,
        help="Number of EC2 instances to use in parallel (default: 1). "
             "Requires multiple entries in eval_config.json for the selected ISA tier.",
    )

    # Eval options
    parser.add_argument("--max-turns", type=int, default=20,
                        help="Max agent turns per definition (default: 20)")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-turn output")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save results to agent-runs/")
    parser.add_argument("--save-trace", action="store_true",
                        help="Include full version_history in result.json")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Skip lazy baseline collection (use if baselines are already present)")

    args = parser.parse_args()
    _start = time.time()

    # ── Resolve ISA → instance type ────────────────────────────────────────
    isa = args.isa or "sve2"
    instance_type = ISA_INSTANCE_MAP.get(isa, "c8g.large")

    if args.provision:
        handles = provision_n(args.num_instances, instance_type, dataset=args.dataset)
    else:
        handles = get_running_instances(isa, args.num_instances)
        if not handles:
            print(f"No running {instance_type} instance(s) for {isa}. Provisioning...")
            handles = provision_n(args.num_instances, instance_type, dataset=args.dataset)
        elif len(handles) < args.num_instances:
            print(
                f"[WARNING] Requested {args.num_instances} instance(s) but only "
                f"{len(handles)} configured in eval_config.json for {isa!r}."
            )

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

    print(
        f"Running {len(problem_defs)} definition(s) across {len(handles)} instance(s) "
        f"(dataset: {args.dataset}, model: {args.model}, instance: {instance_type})"
    )

    # ── Lazy baseline collection ──────────────────────────────────────────
    baseline_author = _DATASET_BASELINE_AUTHOR.get(args.dataset, "reference-scalar")
    if not args.skip_baselines:
        _ensure_baselines_parallel(handles, problem_defs, baseline_author, verbose=not args.quiet)

    # bench_cfg carries the correct baseline_author for this dataset so that
    # AgentTools (and DefaultEvaluator on the remote) use the same author when
    # computing speedup — otherwise speedup is always None.
    bench_cfg = BenchmarkConfig(baseline_author=baseline_author)

    # ── Create pipeline + run evaluations ─────────────────────────────────
    author = args.model.split("/")[-1]
    pipeline = BenchPipeline(handles)

    # Single ThreadPoolExecutor path: max_workers=1 is equivalent to the old
    # sequential for-loop. threading.local() in BenchPipeline assigns handles[0]
    # to the single worker thread when N=1.
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=len(handles)) as pool:
        futures = {
            pool.submit(
                run_agentic_eval,
                defn,
                trace_set=ts,
                author=author,
                model=args.model,
                pipeline=pipeline,
                dataset=args.dataset,
                bench_cfg=bench_cfg,
                max_turns=args.max_turns,
                verbose=not args.quiet,
            ): defn
            for defn in problem_defs
        }

        for i, fut in enumerate(as_completed(futures)):
            defn = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                print(f"\n  ERROR ({defn.name}): {e}")
                result = {
                    "status": "ERROR",
                    "error": str(e),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "version_history": [],
                }
            results[defn.name] = result
            _save_result(defn, result, args)

    # ── Print summary ─────────────────────────────────────────────────────
    _print_summary(results, args.dataset, args.model)
    elapsed = time.time() - _start
    print(f"  Total elapsed:            {elapsed:.1f}s ({elapsed/60:.1f}m)")

    # ── Teardown ──────────────────────────────────────────────────────────
    if args.teardown:
        # Terminate boto3-launched extra instances (handles[1:]) before Terraform
        # destroy, which only manages the primary (handles[0]).
        extra_ids = [h.instance_id for h in handles[1:] if h.instance_id]
        if extra_ids:
            print(f"\n[teardown] Terminating {len(extra_ids)} extra instance(s) via boto3...")
            import boto3
            boto3.client("ec2").terminate_instances(InstanceIds=extra_ids)
        print("\n[teardown] Destroying primary instance via Terraform...")
        teardown()
    else:
        running_hosts = [h.host for h in handles if h.host]
        if running_hosts:
            print(
                f"\n[WARNING] Instance(s) at {running_hosts} are still running and accruing cost. "
                f"Run with --teardown to destroy after evaluation, "
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
