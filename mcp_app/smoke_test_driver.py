"""mcp_app.smoke_test_driver — sequential, non-nanobot fallback/testing driver.

Loosely mirrors the *shape* of eval/run_benchmark.py's per-definition loop
(not its code — zero imports from eval/, see mcp_app/README.md) and is
functionally similar to what skills/nanobot/nanobot-kernel-session/scripts/
launch_session.py does for a real nanobot session, but independently
implemented — no import from skills/ either, so mcp_app and skills/ never
depend on each other. Smoke-test-only — never invoked by the skill.

Formerly split across two files (`driver.py` + `scripts/smoke_test.py`);
merged since `scripts/smoke_test.py` was a strict subset of what this file
already does (a fixed 2-definition sweep with none of the dataset-readiness/
baseline-collection preflight below) — same tool, one name, renamed to make
its smoke-test-only role explicit rather than the more general-sounding
"driver".

Usage:
    # General: any dataset/definition(s).
    python -m mcp_app.smoke_test_driver --host 1.2.3.4 --user ubuntu --key-file ~/.ssh/id_rsa \\
        --dataset ncnn --baseline-author baseline-ncnn-arm --isa sve2 \\
        --problem conv2d_fp32_kh1_kw1_sh1_sw1_dh1_dw1_p0

    # The two definitions used as this repo's standard verification check
    # (see mcp_app/README.md's Verification section):
    python -m mcp_app.smoke_test_driver --host <ip> --user ubuntu --key-file ~/.ssh/id_rsa \\
        --dataset ncnn --baseline-author baseline-ncnn-arm --isa sve2 \\
        --problem conv2d_fp32_kh1_kw1_sh1_sw1_dh1_dw1_p0
    python -m mcp_app.smoke_test_driver --host <ip> --user ubuntu --key-file ~/.ssh/id_rsa \\
        --dataset llama.cpp --baseline-author baseline-llamacpp-arm --isa sve2 \\
        --problem gemm_bf16_n1024_k2048

    python -m mcp_app.smoke_test_driver --list-datasets
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from bench.data.trace_set import TraceSet

from .scripts import _local_ssh
from .scripts.test_mcp_client import run_stdio_sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_TRACE = REPO_ROOT / "bench-trace"

# Directory to mcp_app's own copy dataset_builds.json's content 
DATASET_BUILDS: dict = json.loads((Path(__file__).parent / "dataset_builds.json").read_text())

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


# ── Dataset native-library readiness ────────────────────────────────

def ensure_dataset_ready(
    host: str, user: str, key_file: str, remote_root: str, dataset: str,
    *, verbose: bool = True,
) -> None:
    """Make sure `dataset`'s native-library build artifacts exist on the
    instance, building them if needed. No-op for datasets with no entry in
    DATASET_BUILDS (e.g. simd-loop, which needs no native library). Raises
    RuntimeError if the build steps fail (or the ready_check still fails
    afterward) — callers should not proceed to baseline collection against
    an instance missing what it needs.
    """
    config = DATASET_BUILDS.get(dataset)
    if not config:
        return

    def _ready() -> bool:
        rc, _, _ = _local_ssh.run_remote(host, user, key_file, config["ready_check"], timeout=15)
        return rc == 0

    if _ready():
        if verbose:
            print(f"[dataset] {dataset!r} already built on {host}.")
        return

    if verbose:
        print(f"[dataset] {dataset!r} not ready on {host}; building "
              f"({len(config['steps'])} step(s), this can take several minutes)...")
    for step in config["steps"]:
        if verbose:
            print(f"[dataset]   {step['label']}...")
        rc, _, err = _local_ssh.run_remote(
            host, user, key_file, step["cmd"], timeout=step.get("timeout", 300),
        )
        if rc != 0 and verbose:
            print(f"[dataset]   WARNING: {step['label']} failed: {err[:200]}")

    if not _ready():
        raise RuntimeError(
            f"Dataset {dataset!r} failed to build on {host}. "
            f"SSH in and check manually before running a session (ready_check: {config['ready_check']!r})."
        )
    if verbose:
        print(f"[dataset] {dataset!r} ready on {host}.")


# ── Lazy baseline auto-collection ────────────────────────────────────

def _find_missing_baselines(
    host: str, user: str, key_file: str, remote_root: str,
    definitions: list, baseline_author: str,
) -> list[str]:
    if not definitions:
        return []
    # Requires a PASSED trace, not just any trace with a matching solution
    # prefix — a COMPILE_ERROR/RUNTIME_ERROR baseline trace still "exists"
    # but doesn't satisfy the requirement (exactly what a missing
    # native-library build produces — see ensure_dataset_ready above).
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
        "                    if not line: continue\n"
        "                    rec = json.loads(line)\n"
        "                    if (rec.get('solution', '').startswith(auth)\n"
        "                            and (rec.get('evaluation') or {}).get('status') == 'PASSED'):\n"
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
    *, session_timeout_s: float = 1800,
) -> dict:
    # Still one fresh server process per definition here (compatibility-only
    # update, not the one-server-per-dataset efficiency refactor — see
    # mcp_app/README.md's Open Items). run_dir is author-scoped to match the
    # server's new contract; since each of these spawns only ever touches one
    # definition, its agent-runs-mcp/<author>/<definition_name>/ subdirectory
    # never collides with another spawn's.
    remote_cmd = (
        f"cd {remote_root} && python3 -m mcp_app.server --dataset {dataset} "
        f"--author {author} --baseline-author {baseline_author} --isa {isa} "
        f"--run-dir {remote_root}/agent-runs-mcp/{author} --transport stdio"
    )
    spawn_args = _local_ssh.ssh_spawn_args(host, user, key_file, remote_cmd)
    try:
        result = asyncio.run(asyncio.wait_for(
            run_stdio_sequence("ssh", spawn_args, definition_name, verbose=True),
            timeout=session_timeout_s,
        ))
    except asyncio.TimeoutError:
        # The remote mcp_app.server process can die mid-session
        # without the local ssh child ever seeing a clean disconnect, in
        # which case run_stdio_sequence awaits a response that never comes.
        # Without this, run_definition (and the whole batch loop) hangs
        # forever instead of reporting a failure and moving on.
        result = {
            "status": "TIMEOUT",
            "error": f"no response from remote session within {session_timeout_s}s "
                     "(remote mcp_app.server likely died mid-session)",
        }
    except Exception as e:
        # Covers both explicit correctness-check AssertionErrors from
        # run_tool_sequence and connection-death errors — e.g.
        # mcp.shared.exceptions.McpError (wrapped in an anyio
        # ExceptionGroup) when the ssh session dies mid-call, observed in
        # practice when an unattended-upgrades-triggered sshd restart (or
        # the remote mcp_app.server process exiting unexpectedly) drops the
        # connection. Previously only `except AssertionError` was caught
        # here, so McpError crashed the whole batch loop with an unhandled
        # traceback instead of recording this one definition as failed and
        # continuing to the next.
        result = {"status": "FAILED", "error": str(e)}
    _local_ssh.rsync_from(
        host, user, key_file,
        f"{remote_root}/agent-runs-mcp/{author}/{definition_name}",
        REPO_ROOT / "agent-runs-nanobot",
    )
    return {"definition": definition_name, **result}


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
    p.add_argument("--skip-dataset-check", action="store_true",
                    help="Skip ensure_dataset_ready (ncnn/llama.cpp native-library build check).")
    p.add_argument("--baseline-parallelism", type=int, default=4)
    p.add_argument("--session-timeout-s", type=float, default=1800,
                    help="Give up on a single definition's MCP session (compile/evaluate/"
                         "disassemble/submit) after this many seconds with no response, "
                         "reporting status=TIMEOUT instead of hanging forever.")
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

    # Sync first — ensure_dataset_ready/ensure_baselines both need bench-trace/
    # and bench/ to already exist at remote_root (the baseline-presence check
    # script reads remote_root/bench-trace/traces/; 
    _local_ssh.rsync_to(
        args.host, args.user, args.key_file, REPO_ROOT, args.remote_root,
        excludes=DEFAULT_RSYNC_EXCLUDES,
    )

    #collect-baselines needs bench/ importable there too). Running this before the sync used to be
    # a latent ordering bug that just didn't manifest if the instance
    # happened to already be synced from an earlier run.
    if not args.skip_dataset_check:
        ensure_dataset_ready(args.host, args.user, args.key_file, args.remote_root, args.dataset)

    if not args.skip_baselines:
        ensure_baselines(
            args.host, args.user, args.key_file, args.remote_root,
            problem_defs, args.baseline_author, parallelism=args.baseline_parallelism,
        )

    results = []
    for i, defn in enumerate(problem_defs):
        print(f"\n[{i + 1}/{len(problem_defs)}] {defn.name}")
        start = time.monotonic()
        result = run_definition(
            args.host, args.user, args.key_file, args.remote_root,
            defn.name, args.dataset, args.author, args.baseline_author, args.isa,
            session_timeout_s=args.session_timeout_s,
        )
        result["duration_s"] = round(time.monotonic() - start, 1)
        results.append(result)
        print(f"  -> {result['status']} ({result['duration_s']}s)")

    n_ok = sum(1 for r in results if r["status"] == "PASSED")
    print(f"\n{'=' * 60}\nDone: {n_ok}/{len(results)} succeeded\n{'=' * 60}")


if __name__ == "__main__":
    main()
