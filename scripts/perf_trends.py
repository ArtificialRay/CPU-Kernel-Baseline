"""
scripts/perf_trends.py — Show benchmark performance trends over time.

Reads append-only JSONL result files from results/ and prints a time-series
table of speedup metrics per (problem, isa, model).

Usage:
    # All problems/models:
    python scripts/perf_trends.py

    # Filter to a specific model:
    python scripts/perf_trends.py --model anthropic/claude-sonnet-4-6

    # Filter to a specific problem:
    python scripts/perf_trends.py --problem loop_001

    # Show per-turn tool usage summary from a trace file:
    python scripts/perf_trends.py --trace traces/loop_001_sve_anthropic_claude-sonnet-4-6_2026-04-14T12-00-00Z.json
"""

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def print_trends(model_filter: str | None, problem_filter: str | None):
    jsonl_files = sorted(RESULTS_DIR.glob("*.jsonl"))
    if not jsonl_files:
        print("No .jsonl result files found in results/. Run the benchmark with at least one eval first.")
        return

    # Group records by (problem_id, isa, model)
    groups: dict[tuple, list[dict]] = {}
    for path in jsonl_files:
        records = load_jsonl(path)
        for r in records:
            pid = r.get("problem_id", "")
            isa = r.get("isa", "")
            model = r.get("model", "")
            if model_filter and model_filter not in model:
                continue
            if problem_filter and problem_filter not in pid:
                continue
            key = (pid, isa, model)
            groups.setdefault(key, []).append(r)

    if not groups:
        print("No matching records found.")
        return

    for (pid, isa, model), records in sorted(groups.items()):
        # Sort by timestamp
        records.sort(key=lambda r: r.get("timestamp", ""))

        print(f"\n{'─'*72}")
        print(f"  {pid}  |  isa={isa}  |  model={model}")
        print(f"{'─'*72}")
        header = f"  {'Timestamp':<22}  {'Lvl':>3}  {'vs_scalar':>10}  {'vs_autovec':>10}  {'vs_ref':>8}  {'tools':>5}  {'correct'}"
        print(header)
        print(f"  {'-'*68}")

        for r in records:
            ts = r.get("timestamp", "")[:19]
            lvl = r.get("level", "?")
            vs_s = r.get("speedup_vs_scalar")
            vs_a = r.get("speedup_vs_autovec")
            vs_r = r.get("speedup_vs_ref")
            tools = r.get("tool_calls", "?")
            correct = "✓" if r.get("correct") else "✗"

            def fmt(v):
                return f"{v:.2f}×" if v is not None else "    —  "

            print(f"  {ts:<22}  {lvl:>3}  {fmt(vs_s):>10}  {fmt(vs_a):>10}  {fmt(vs_r):>8}  {tools:>5}  {correct}")

        # Trend summary if ≥2 runs
        correct_runs = [r for r in records if r.get("correct")]
        if len(correct_runs) >= 2:
            first, last = correct_runs[0], correct_runs[-1]
            delta = (last.get("speedup_vs_scalar") or 0) - (first.get("speedup_vs_scalar") or 0)
            direction = "▲" if delta > 0 else ("▼" if delta < 0 else "━")
            print(f"\n  Trend (first→last correct run): speedup_vs_scalar {direction} {delta:+.2f}×")

    print(f"\n{'─'*72}")
    print(f"  Total groups: {len(groups)}")


def print_trace(trace_path: Path):
    data = json.loads(trace_path.read_text())
    print(f"\nTrace: {data.get('problem_id')} | isa={data.get('isa')} | model={data.get('model')}")
    print(f"Timestamp: {data.get('timestamp')}")

    # Code version summary table
    versions = data.get("code_versions", [])
    correct_versions = [v for v in versions if v.get("correct")]
    if versions:
        print(f"\n{'─'*72}")
        print(f"  Code versions ({len(versions)} compiled, {len(correct_versions)} correct)")
        print(f"  {'v':>3}  {'turn':>4}  {'ms/iter':>9}  {'IPC':>6}  {'task_clk':>10}  notes")
        print(f"  {'─'*60}")
        best_ms = min((v["ms_per_iter"] for v in correct_versions if v.get("ms_per_iter")), default=None)
        for v in versions:
            ms = v.get("ms_per_iter")
            perf = v.get("perf") or {}
            ipc = perf.get("ipc", "")
            tc = perf.get("task_clock_ms", "")
            ms_str = f"{ms:.4f}" if ms is not None else "    —  "
            ipc_str = f"{ipc:.2f}" if ipc else "    —"
            tc_str = f"{tc:.4f}" if tc else "      —  "
            notes = ""
            if not v.get("correct"):
                notes = "incorrect"
            elif ms is not None and ms == best_ms:
                notes = "← BEST"
            print(f"  {v['version']:>3}  {v['turn']:>4}  {ms_str:>9}  {ipc_str:>6}  {tc_str:>10}  {notes}")

    print(f"\n{'─'*72}")
    print("  Per-turn reasoning:")

    for entry in data.get("trace", []):
        turn = entry.get("turn")
        tool = entry.get("tool")
        reasoning = (entry.get("reasoning") or "").strip()
        code_diff = entry.get("code_diff")
        args = entry.get("args", {})
        result = entry.get("result", {})

        print(f"\n{'─'*72}")
        print(f"[Turn {turn}] → {tool}")

        if reasoning:
            print(f"  Reasoning:")
            for line in reasoning.splitlines():
                print(f"    {line}")

        if code_diff:
            print(f"  Code diff (vs previous version):")
            for line in code_diff.splitlines():
                prefix = "  "
                if line.startswith("+"):
                    prefix = "  \033[32m"   # green
                elif line.startswith("-"):
                    prefix = "  \033[31m"   # red
                print(f"{prefix}  {line}\033[0m")

        # Args summary (skip large code, show just first line + length)
        for k, v in args.items():
            if k == "code":
                first_line = str(v).splitlines()[0][:80] if v else ""
                print(f"  code: {first_line!r}… [{len(str(v))} chars]")
            elif k != "explanation":
                print(f"  arg.{k}: {v!r}")

        # Result summary
        result_parts = []
        for k, v in result.items():
            if v is not None:
                result_parts.append(f"{k}={v}")
        if result_parts:
            print(f"  result: {', '.join(result_parts)}")


def main():
    parser = argparse.ArgumentParser(description="Show benchmark performance trends over time")
    parser.add_argument("--model", default=None, help="Filter by model substring")
    parser.add_argument("--problem", default=None, help="Filter by problem ID substring")
    parser.add_argument("--trace", default=None, help="Path to a trace JSON file to inspect")
    args = parser.parse_args()

    if args.trace:
        print_trace(Path(args.trace))
    else:
        print_trends(args.model, args.problem)


if __name__ == "__main__":
    main()
