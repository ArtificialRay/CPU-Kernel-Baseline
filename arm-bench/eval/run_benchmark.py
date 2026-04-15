"""
eval/run_benchmark.py — Agentic benchmark CLI for ncnn kernel optimization.

Full end-to-end: provision (if needed) -> run agentic LLM eval -> score -> (optionally) teardown.

Usage (from arm-bench/), using openrouter:
    # Single problem, local:
    python -m eval.run_benchmark --problem conv2d --isa sve --model openrouter/anthropic/claude-opus-4-6

    # All problems:
    python -m eval.run_benchmark --all --isa sve --model openrouter/anthropic/claude-opus-4-6

    # With remote instance:
    python -m eval.run_benchmark --all --isa sve --model openrouter/anthropic/claude-opus-4-6 \
        --provision --teardown
"""

import argparse
import json

from eval.config import REPO_ROOT
from eval.evaluator import run_agentic_eval, load_starter_problems
from eval.tools import EvalResult

RESULTS_DIR = REPO_ROOT / "results"


def main():
    parser = argparse.ArgumentParser(
        description="Agentic LLM benchmark for ncnn kernel optimization"
    )

    # Problem selection
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--problem", help="Single problem ID, e.g. conv2d")
    grp.add_argument("--all", action="store_true", help="Run all problems for the given ISA")

    # Model and ISA
    parser.add_argument("--isa", required=True, choices=["neon", "sve", "sve2"])
    parser.add_argument("--model", required=True,
                        help="LiteLLM model string, e.g. anthropic/claude-opus-4-6")

    # Instance lifecycle
    parser.add_argument("--provision", action="store_true",
                        help="Provision/reuse a remote instance instead of running locally")
    parser.add_argument("--teardown", action="store_true",
                        help="Destroy the instance after evaluation")

    # Eval options
    parser.add_argument("--max-turns", type=int, default=20,
                        help="Max agent turns per problem (default: 20)")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-turn output")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to results/")

    args = parser.parse_args()

    # ── Resolve instance ──────────────────────────────────────────────────
    handle = None
    if args.provision:
        from eval.provision import get_or_provision
        handle = get_or_provision(args.isa)
        print(f"Using remote instance: {handle.host}")

    # ── Resolve problems ──────────────────────────────────────────────────
    problems = load_starter_problems()

    if args.problem:
        if args.problem not in problems:
            print(f"Problem {args.problem!r} not found in starter/problems.json")
            return
        problem_ids = [args.problem]
    else:
        problem_ids = [
            pid for pid, p in problems.items()
            if p.get("isa_target") == args.isa
        ]
        print(f"Running {len(problem_ids)} problems (ISA: {args.isa})")

    # ── Run evaluations ──────────────────────────────────────────────────
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

    # ── Print summary ────────────────────────────────────────────────────
    _print_summary(results, args.isa, args.model)

    # ── Teardown ─────────────────────────────────────────────────────────
    if args.teardown:
        from eval.provision import teardown
        print("\n[teardown] Destroying instance...")
        teardown()


def _print_summary(results: dict[str, EvalResult], isa: str, model: str):
    n = len(results)
    if n == 0:
        return

    n_correct = sum(1 for r in results.values() if r.correct)
    n_beats_baseline = sum(1 for r in results.values() if r.level >= 2)
    speedups = [r.speedup_vs_ref for r in results.values() if r.speedup_vs_ref is not None]
    avg_speedup = round(sum(speedups) / len(speedups), 2) if speedups else None

    print(f"\n{'='*60}")
    print(f"  Benchmark Summary")
    print(f"  Model: {model}  |  ISA: {isa}")
    print(f"{'='*60}")
    print(f"  Total problems:               {n}")
    print(f"  Last compiled code Correctness (level >= 1):          {n_correct}/{n}")
    print(f"  Beats ARM baseline (level 2):  {n_beats_baseline}/{n}")
    if avg_speedup is not None:
        print(f"  Avg speedup vs ARM baseline:   {avg_speedup}x")
    print(f"{'='*60}")

    # Per-problem table
    print(f"\n{'Problem':<20} {'Correct(last compiled code)':<10} {'Speedup(best version)':<12} {'Runtime ms':<12} {'Turns'} {'Metadata location for code with best speedup'}")
    print("-" * 66)
    for pid, r in results.items():
        correct = "PASS" if r.correct else "FAIL"
        speedup = f"{r.speedup_vs_ref}x" if r.speedup_vs_ref else "N/A"
        ms = str(r.runtime_ms) if r.runtime_ms else "N/A"
        turns = str(r.tool_calls)
        print(f"{pid:<20} {correct:<10} {speedup:<12} {ms:<12} {turns} {r.best_history_file}")


if __name__ == "__main__":
    main()