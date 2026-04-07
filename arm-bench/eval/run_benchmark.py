"""
eval/run_benchmark.py — Agentic benchmark CLI for simd-loops.

Full end-to-end: provision (if needed) → run agentic LLM eval → score → (optionally) teardown.

Usage:
    # Single problem, agentic mode:
    python eval/run_benchmark.py --problem loop_001 --isa sve2 --model anthropic/claude-opus-4-6

    # Full benchmark (all problems for an ISA):
    python eval/run_benchmark.py --all --isa sve2 --model anthropic/claude-opus-4-6

    # Provision a fresh instance, run, then tear it down:
    python eval/run_benchmark.py --problem loop_001 --isa sve2 --model anthropic/claude-opus-4-6 \\
        --provision --teardown

    # Use an already-running instance (from eval_config.json):
    python eval/run_benchmark.py --all --isa sve --model openai/gpt-4o
"""

import argparse
import json
import time
from pathlib import Path

from eval.config import REPO_ROOT, load_problems, ISA_TIER
from eval.evaluator import run_agentic_eval
from eval.provision import get_or_provision, get_running_instance, teardown, provision, ISA_INSTANCE_MAP
from eval.tools import EvalResult

RESULTS_DIR = REPO_ROOT / "results"


def main():
    parser = argparse.ArgumentParser(
        description="Agentic LLM benchmark for simd-loops SIMD kernels"
    )

    # Problem selection
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--problem", help="Single problem ID, e.g. loop_001")
    grp.add_argument("--all", action="store_true", help="Run all problems for the given ISA")

    # Model and ISA
    parser.add_argument("--isa", required=True, choices=["neon", "sve", "sve2", "sme2"])
    parser.add_argument("--model", required=True,
                        help="LiteLLM model string, e.g. anthropic/claude-opus-4-6")

    # Instance lifecycle
    parser.add_argument("--provision", action="store_true",
                        help="Provision a new instance even if one is already configured")
    parser.add_argument("--teardown", action="store_true",
                        help="Destroy the instance after evaluation")
    parser.add_argument("--instance", default=None,
                        help="EC2 instance type override (default: inferred from ISA)")

    # Eval options
    parser.add_argument("--max-turns", type=int, default=20,
                        help="Max agent turns per problem (default: 20)")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-turn output")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to results/")

    args = parser.parse_args()

    # SME2 requires hardware not yet available on AWS (no EC2 instance supports SME2).
    # The problems are kept in the dataset for future use.
    if args.isa == "sme2":
        print("ERROR: SME2 is not yet supported — no AWS EC2 instance implements the "
              "Scalable Matrix Extension. SME2 problems are reserved for future hardware. "
              "Use --isa sve (c7g) or --isa sve2 (c8g) instead.")
        return

    # ── Resolve instance ───────────────────────────────────────────────────
    instance_type = args.instance or ISA_INSTANCE_MAP.get(args.isa, "c7g.large")

    if args.provision:
        handle = provision(instance_type)
    else:
        handle = get_running_instance(args.isa)
        if handle is None:
            print(f"No running instance for ISA={args.isa}. Provisioning {instance_type}...")
            handle = provision(instance_type)

    # ── Resolve problems ──────────────────────────────────────────────────
    problems = load_problems()
    if args.problem:
        if args.problem not in problems:
            print(f"Problem {args.problem!r} not found in problems.json")
            return
        problem_ids = [args.problem]
    else:
        problem_ids = [
            pid for pid, p in problems.items()
            if p.get("isa_target") == args.isa
        ]
        print(f"Running {len(problem_ids)} problems (ISA: {args.isa})")

    # ── Run evaluations ───────────────────────────────────────────────────
    results: dict[str, EvalResult] = {}
    RESULTS_DIR.mkdir(exist_ok=True)

    for i, pid in enumerate(problem_ids):
        print(f"\n[{i+1}/{len(problem_ids)}] {pid}")
        try:
            result = run_agentic_eval(
                problem_id=pid,
                isa=args.isa,
                model=args.model,
                handle=handle,
                max_turns=args.max_turns,
                verbose=not args.quiet,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            result = EvalResult(correct=False, level=0, compile_error=str(e))

        results[pid] = result

        if not args.no_save:
            out = RESULTS_DIR / f"{pid}_{args.isa}_{args.model.replace('/', '_')}.json"
            data = result.to_dict()
            data.update({"problem_id": pid, "isa": args.isa, "model": args.model})
            out.write_text(json.dumps(data, indent=2))

    # ── Print summary ─────────────────────────────────────────────────────
    _print_summary(results, args.isa, args.model)

    # ── Teardown ──────────────────────────────────────────────────────────
    if args.teardown:
        print("\n[teardown] Destroying instance...")
        teardown()


def _print_summary(results: dict[str, EvalResult], isa: str, model: str):
    n = len(results)
    if n == 0:
        return

    by_level = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    speedups = []
    for r in results.values():
        by_level[r.level] = by_level.get(r.level, 0) + 1
        if r.speedup_vs_scalar is not None:
            speedups.append(r.speedup_vs_scalar)

    n_correct = sum(v for k, v in by_level.items() if k >= 1)
    n_fast = sum(v for k, v in by_level.items() if k >= 2)
    n_autovec = sum(v for k, v in by_level.items() if k >= 3)
    n_ref = by_level.get(4, 0)
    avg_speedup = round(sum(speedups) / len(speedups), 2) if speedups else None

    print(f"\n{'='*60}")
    print(f"  Benchmark Summary")
    print(f"  Model: {model}  |  ISA: {isa}")
    print(f"{'='*60}")
    print(f"  Total problems:           {n}")
    print(f"  Correct (level ≥ 1):      {n_correct}/{n}  ({100*n_correct//n}%)")
    print(f"  Beats scalar (level ≥ 2): {n_fast}/{n}  ({100*n_fast//n}%)  ← fast_p")
    print(f"  Beats autovec (level ≥ 3): {n_autovec}/{n}  ({100*n_autovec//n}%)")
    print(f"  Beats hand-written (level 4): {n_ref}/{n}  ({100*n_ref//n}%)")
    if avg_speedup is not None:
        print(f"  Avg speedup vs scalar:    {avg_speedup}×  (correct submissions)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
