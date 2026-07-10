"""mcp_app.driver — sequential, non-nanobot fallback/testing driver.

Loosely mirrors the *shape* of eval/run_benchmark.py's per-definition loop
(not its code — zero imports from eval/, see mcp_app/README.md) and is
functionally similar to what skills/nanobot/nanobot-kernel-session/scripts/
launch_session.py does for a real nanobot session, but independently
implemented — no import from skills/ either, so mcp_app and skills/ never
depend on each other.

Usage:
    python -m mcp_app.driver --host 1.2.3.4 --user ubuntu --key-file ~/.ssh/id_rsa \\
        --dataset ncnn --baseline-author baseline-ncnn-arm --isa sve2 \\
        --problem conv2d_fp32_kh1_kw1_sh1_sw1_dh1_dw1_p0

    python -m mcp_app.driver --list-datasets
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from bench.data.trace_set import TraceSet

from .scripts import _local_ssh
from .scripts.test_mcp_client import run_stdio_sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_TRACE = REPO_ROOT / "bench-trace"

DEFAULT_RSYNC_EXCLUDES = [
    "build", ".git", "terraform", "generations", "results", "notebooks",
    "agent-runs", "agent-runs-mcp", "agent-runs-nanobot", "__pycache__", "*.pyc",
]

# Hand-typed, matching skills/nanobot/nanobot-kernel-session/SKILL.md's table
# (decision 10 — no shared source of truth to import from eval/run_benchmark.py).
DATASET_REFERENCE = {
    "ncnn": {"baseline_author": "baseline-ncnn-arm", "isa_hint": "sve (Graviton3) / sve2 (Graviton4)"},
    "simd-loop": {"baseline_author": "reference", "isa_hint": "sve (Graviton3) / sve2 (Graviton4)"},
    "llama.cpp": {"baseline_author": "baseline-llamacpp-arm", "isa_hint": "sve (Graviton3) / sve2 (Graviton4)"},
}


def _print_dataset_reference() -> None:
    print(f"{'dataset':<12} {'baseline_author':<22} isa")
    for dataset, info in DATASET_REFERENCE.items():
        print(f"{dataset:<12} {info['baseline_author']:<22} {info['isa_hint']}")


def _defs_for_dataset(ts: TraceSet, dataset: str) -> list:
    """Definitions that have at least one solution in the given dataset."""
    matching = {
        def_name
        for def_name, sols in ts.solutions.items()
        if any(s.dataset.value == dataset for s in sols)
    }
    return [ts.definitions[n] for n in sorted(matching) if n in ts.definitions]


# ── Lazy baseline auto-collection (adapted from eval/run_benchmark.py's
#    _ensure_baselines — see mcp_app/README.md on why this is a copy, not an
#    import) — parallelized across missing definitions, safe because
#    BuilderRegistry's build cache is lock-serialized per (solution.hash(),
#    is_baseline), so concurrent builds of *different* definitions never
#    contend (bench/compile/registry.py). ────────────────────────────────────

def _find_missing_baselines(
    host: str, user: str, key_file: str, remote_root: str,
    definitions: list, baseline_author: str,
) -> list[str]:
    if not definitions:
        return []
    check_code = (
        "import json; from pathlib import Path\n"
        f"bench = Path({remote_root!r}).expanduser() / 'bench-trace'\n"
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
    _, out, _ = _local_ssh.run_remote(
        host, user, key_file, f"echo {b64!r} | base64 -d | python3", timeout=30,
    )
    return out.split()


def ensure_baselines(
    host: str, user: str, key_file: str, remote_root: str,
    definitions: list, baseline_author: str,
    *, parallelism: int = 4, verbose: bool = True,
) -> None:
    missing_names = set(_find_missing_baselines(
        host, user, key_file, remote_root, definitions, baseline_author,
    ))
    missing = [d for d in definitions if d.name in missing_names]
    if not missing:
        if verbose:
            print(f"[baselines] All {len(definitions)} baseline trace(s) present.")
        return

    if verbose:
        print(
            f"[baselines] {len(missing)}/{len(definitions)} definition(s) missing baseline "
            f"traces — collecting in parallel (parallelism={parallelism}, "
            f"author={baseline_author!r})..."
        )

    def _collect_one(d) -> tuple[str, int, str, str]:
        rc, out, err = _local_ssh.run_remote(
            host, user, key_file,
            f"cd {remote_root} && python3 -m bench.cli collect-baselines "
            f"--baseline-author {baseline_author} --definition {d.name}",
            timeout=600,
        )
        return d.name, rc, out, err

    with ThreadPoolExecutor(max_workers=parallelism) as pool:
        futures = {pool.submit(_collect_one, d): d for d in missing}
        for future in as_completed(futures):
            name, rc, out, err = future.result()
            if verbose:
                if rc == 0:
                    print(f"  [baselines] {name}: OK")
                else:
                    combined = "\n".join(filter(None, [out.strip(), err.strip()]))
                    print(f"  [baselines] {name}: WARNING {combined}")


# ── Per-definition session loop ────────────────────────────────────────────

def run_definition(
    host: str, user: str, key_file: str, remote_root: str,
    definition_name: str, dataset: str, author: str, baseline_author: str, isa: str,
) -> dict:
    remote_cmd = (
        f"cd {remote_root} && python3 -m mcp_app.server --dataset {dataset} "
        f"--definition {definition_name} --author {author} "
        f"--baseline-author {baseline_author} --isa {isa} "
        f"--run-dir {remote_root}/agent-runs-mcp/{definition_name} --transport stdio"
    )
    spawn_args = _local_ssh.ssh_spawn_args(host, user, key_file, remote_cmd)
    try:
        result = asyncio.run(run_stdio_sequence("ssh", spawn_args, verbose=True))
        status = "ok"
    except AssertionError as e:
        result = {"error": str(e)}
        status = "failed"
    _local_ssh.rsync_from(
        host, user, key_file,
        f"{remote_root}/agent-runs-mcp/{definition_name}",
        REPO_ROOT / "agent-runs-nanobot",
    )
    return {"definition": definition_name, "status": status, **result}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list-datasets", action="store_true",
                    help="Print the dataset/baseline_author/isa reference table and exit.")
    p.add_argument("--host")
    p.add_argument("--user", default="ubuntu")
    p.add_argument("--key-file", default="~/.ssh/id_rsa")
    p.add_argument("--remote-root", default="~/arm-bench")
    p.add_argument("--dataset", choices=list(DATASET_REFERENCE))
    p.add_argument("--baseline-author")
    p.add_argument("--isa", choices=["neon", "sve", "sve2", "sme2"])
    p.add_argument("--author", default="driver-test")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--problem", help="Definition name or op_type prefix.")
    grp.add_argument("--all", action="store_true", help="Run all definitions for the dataset.")
    p.add_argument("--skip-baselines", action="store_true")
    p.add_argument("--baseline-parallelism", type=int, default=4)
    p.add_argument("--bench-trace-root", default=str(BENCH_TRACE))
    args = p.parse_args(argv)

    if args.list_datasets:
        _print_dataset_reference()
        return

    for required in ("host", "dataset", "baseline_author", "isa"):
        if getattr(args, required) is None:
            p.error(f"--{required.replace('_', '-')} is required (unless --list-datasets)")
    if not args.problem and not args.all:
        p.error("one of --problem or --all is required")

    ts = TraceSet.from_path(Path(args.bench_trace_root))
    dataset_defs = _defs_for_dataset(ts, args.dataset)
    if args.problem:
        problem_defs = [
            d for d in dataset_defs
            if d.op_type == args.problem or d.name.startswith(args.problem)
        ]
        if not problem_defs:
            available = sorted({d.op_type for d in dataset_defs})
            print(f"No definitions matching {args.problem!r} in dataset {args.dataset!r}.")
            print(f"Available op_types: {available}")
            sys.exit(1)
    else:
        problem_defs = dataset_defs

    print(f"Running {len(problem_defs)} definition(s) (dataset={args.dataset}, isa={args.isa})")

    if not args.skip_baselines:
        ensure_baselines(
            args.host, args.user, args.key_file, args.remote_root,
            problem_defs, args.baseline_author, parallelism=args.baseline_parallelism,
        )

    _local_ssh.rsync_to(
        args.host, args.user, args.key_file, REPO_ROOT, args.remote_root,
        excludes=DEFAULT_RSYNC_EXCLUDES,
    )

    results = []
    for i, defn in enumerate(problem_defs):
        print(f"\n[{i + 1}/{len(problem_defs)}] {defn.name}")
        start = time.monotonic()
        result = run_definition(
            args.host, args.user, args.key_file, args.remote_root,
            defn.name, args.dataset, args.author, args.baseline_author, args.isa,
        )
        result["duration_s"] = round(time.monotonic() - start, 1)
        results.append(result)
        print(f"  -> {result['status']} ({result['duration_s']}s)")

    n_ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n{'=' * 60}\nDone: {n_ok}/{len(results)} succeeded\n{'=' * 60}")


if __name__ == "__main__":
    main()
