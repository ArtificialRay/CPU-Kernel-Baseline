"""launch_session — bring up one mcp_app session on a remote instance, sync
results back afterward. Runs on the *caller's* host (nanobot's host), not on
the target instance.

Split into two independent, composable steps rather than one process-owning
start/stop pair: in the preferred stdio-over-ssh mode, nanobot's own MCP
client spawns the SSH command itself and owns that subprocess end to end —
this script never holds that process handle, so it can't use its own
lifecycle to know when to sync results back. "When is this session done" is
left to whatever orchestrates nanobot; that layer calls prepare_session()
before starting nanobot and sync_results() after nanobot finishes.

Zero imports from mcp_app or eval/ — see remote.py's module docstring and
mcp_app/README.md's "Scope boundary" section.
"""

from __future__ import annotations

import argparse
import base64
import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from remote import RemoteTarget

DEFAULT_RSYNC_EXCLUDES = [
    "build", ".git", "terraform", "generations", "results", "notebooks",
    "agent-runs", "agent-runs-mcp", "agent-runs-nanobot", "__pycache__", "*.pyc",
]

# Directory to this skill's own copy of eval/dataset_builds.json's content
DATASET_BUILDS: dict = json.loads((Path(__file__).parent / "dataset_builds.json").read_text())


def _spawn_command(
    target: RemoteTarget, remote_root: str, dataset: str, definition_name: str,
    author: str, baseline_author: str, isa: str, *, transport: str, port: Optional[int] = None,
) -> str:
    run_dir = f"{remote_root}/agent-runs-mcp/{definition_name}"
    cmd = (
        f"cd {remote_root} && python3 -m mcp_app.server --dataset {dataset} "
        f"--definition {definition_name} --author {author} "
        f"--baseline-author {baseline_author} --isa {isa} "
        f"--run-dir {run_dir} --transport {transport}"
    )
    if transport == "sse":
        cmd += f" --bind-host 127.0.0.1 --port {port}"
    return cmd


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ── Pre-flight: make sure the instance actually has what a session needs
#    before nanobot ever connects. mcp_app/smoke_test_driver.py (mcp_app's
#    non-nanobot smoke-test tool) has its own independent copy of this same
#    logic — see mcp_app/README.md's Scope boundary on why this isn't shared.

def ensure_dataset_ready(target: RemoteTarget, dataset: str, *, verbose: bool = True) -> None:
    """Make sure `dataset`'s native-library build artifacts (ncnn/llama.cpp)
    exist on the instance, building them if needed. No-op for datasets with
    no entry in DATASET_BUILDS (e.g. simd-loop). Raises RuntimeError if the
    build steps fail (or the ready_check still fails afterward).
    """
    config = DATASET_BUILDS.get(dataset)
    if not config:
        return

    def _ready() -> bool:
        rc, _, _ = target.run(config["ready_check"], timeout=15)
        return rc == 0

    if _ready():
        if verbose:
            print(f"[dataset] {dataset!r} already built on {target.host}.")
        return

    if verbose:
        print(f"[dataset] {dataset!r} not ready on {target.host}; building "
              f"({len(config['steps'])} step(s), this can take several minutes)...")
    for step in config["steps"]:
        if verbose:
            print(f"[dataset]   {step['label']}...")
        rc, _, err = target.run(step["cmd"], timeout=step.get("timeout", 300))
        if rc != 0 and verbose:
            print(f"[dataset]   WARNING: {step['label']} failed: {err[:200]}")

    if not _ready():
        raise RuntimeError(
            f"Dataset {dataset!r} failed to build on {target.host}. "
            f"SSH in and check manually before starting a session (ready_check: {config['ready_check']!r})."
        )
    if verbose:
        print(f"[dataset] {dataset!r} ready on {target.host}.")


def ensure_baseline_collected(
    target: RemoteTarget, remote_root: str, definition_name: str, baseline_author: str,
    *, verbose: bool = True,
) -> None:
    """Make sure `definition_name` has a PASSED baseline trace for
    `baseline_author`, running `bench.cli collect-baselines` for just this
    one definition if not — without this, evaluate()/submit() silently
    return time_speedup=None/cycle_speedup=None (correctness still works,
    only the speedup numbers are missing). Scoped to one definition (not a
    batch like smoke_test_driver.py's version) since this is a
    single-session helper.
    """
    check_code = (
        "import json; from pathlib import Path\n"
        f"bench = Path({remote_root!r}).expanduser() / 'bench-trace'\n"
        f"auth = {baseline_author!r}\n"
        f"name = {definition_name!r}\n"
        "found = False\n"
        "td = bench / 'traces'\n"
        "if td.exists():\n"
        "    for f in sorted(td.rglob(name + '.jsonl')):\n"
        "        try:\n"
        "            for line in f.open():\n"
        "                line = line.strip()\n"
        "                if not line: continue\n"
        "                rec = json.loads(line)\n"
        "                if (rec.get('solution', '').startswith(auth)\n"
        "                        and (rec.get('evaluation') or {}).get('status') == 'PASSED'):\n"
        "                    found = True; break\n"
        "        except Exception:\n"
        "            pass\n"
        "        if found: break\n"
        "print('READY' if found else 'MISSING')\n"
    )
    b64 = base64.b64encode(check_code.encode()).decode()
    _, out, _ = target.run(f"echo {b64!r} | base64 -d | python3", timeout=30)
    if out.strip() == "READY":
        if verbose:
            print(f"[baseline] {definition_name!r} already has a PASSED {baseline_author!r} trace.")
        return

    if verbose:
        print(f"[baseline] Collecting {baseline_author!r} baseline for {definition_name!r}...")
    rc, out, err = target.run(
        f"cd {remote_root} && python3 -m bench.cli collect-baselines "
        f"--baseline-author {baseline_author} --definition {definition_name}",
        timeout=600,
    )
    if rc != 0:
        combined = "\n".join(filter(None, [out.strip(), err.strip()]))
        raise RuntimeError(
            f"collect-baselines failed for {definition_name!r} (author={baseline_author!r}): {combined[:500]}"
        )
    if verbose:
        print(f"[baseline] {definition_name!r}: OK")


def prepare_session(
    target: RemoteTarget,
    definition_name: str,
    dataset: str,
    author: str,
    baseline_author: str,
    isa: str,
    *,
    remote_root: str = "~/arm-bench",
    sync_repo: bool = True,
    local_repo_dir: Optional[str | Path] = None,
    skip_preflight: bool = False,
    transport: str = "stdio",
    local_port: Optional[int] = None,
    remote_port: int = 8765,
    startup_timeout: int = 60,
) -> dict:
    """Get an mcp_app session ready to be driven by a real MCP client.

    Also ensures the instance is actually ready for this session before
    returning (unless skip_preflight=True): the dataset's native library is
    built (ensure_dataset_ready) and this definition has a real PASSED
    baseline trace (ensure_baseline_collected) — both idempotent, so calling
    this repeatedly for the same instance is cheap after the first time.
    mcp_app/smoke_test_driver.py is smoke-test-only and is never invoked by
    this skill; a real nanobot session is fully self-sufficient through this
    one call.

    transport="stdio" (default, try first): returns a spawn command dict
    ({"transport": "stdio", "command": "ssh", "args": [...]}) for nanobot's
    own MCP config — no process is spawned by this function in this mode.

    transport="sse" (fallback, only if stdio doesn't work with nanobot's
    config format): establishes an SSH local-port-forward + starts the
    remote server, returns {"transport": "sse", "endpoint": "http://127.0.0.1:<port>",
    "_tunnel_proc": <Popen>} — call stop_tunnel() on the result when done.
    """
    if sync_repo:
        if local_repo_dir is None:
            raise ValueError("local_repo_dir is required when sync_repo=True")
        target.rsync_to(local_repo_dir, remote_root, excludes=DEFAULT_RSYNC_EXCLUDES)

    if not skip_preflight:
        ensure_dataset_ready(target, dataset)
        ensure_baseline_collected(target, remote_root, definition_name, baseline_author)

    if transport == "stdio":
        remote_cmd = _spawn_command(
            target, remote_root, dataset, definition_name, author, baseline_author, isa,
            transport="stdio",
        )
        return {
            "transport": "stdio",
            "command": "ssh",
            "args": [*target.ssh_base_args(), f"{target.user}@{target.host}", remote_cmd],
        }

    if transport == "sse":
        if local_port is None:
            local_port = _free_local_port()
        remote_cmd = _spawn_command(
            target, remote_root, dataset, definition_name, author, baseline_author, isa,
            transport="sse", port=remote_port,
        )
        ssh_cmd = [
            "ssh", "-L", f"{local_port}:127.0.0.1:{remote_port}",
            *target.ssh_base_args(), f"{target.user}@{target.host}", remote_cmd,
        ]
        proc = subprocess.Popen(ssh_cmd)
        endpoint = f"http://127.0.0.1:{local_port}/sse"
        _wait_for_port(local_port, timeout=startup_timeout, proc=proc)
        return {"transport": "sse", "endpoint": endpoint, "_tunnel_proc": proc}

    raise ValueError(f"Unknown transport: {transport!r}")


def _wait_for_port(port: int, *, timeout: float, proc: subprocess.Popen) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"ssh tunnel process exited early (rc={proc.returncode})")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"Nothing listening on 127.0.0.1:{port} after {timeout}s")


def stop_tunnel(prepared: dict) -> None:
    """Only meaningful for transport="sse" — no-op otherwise."""
    proc = prepared.get("_tunnel_proc")
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def sync_results(
    target: RemoteTarget,
    definition_name: str,
    *,
    remote_root: str = "~/arm-bench",
    local_results_dir: str | Path,
) -> dict:
    """Pull agent-runs-mcp/<definition_name>/ back to local_results_dir/<definition_name>/."""
    target.rsync_from(
        f"{remote_root}/agent-runs-mcp/{definition_name}", local_results_dir,
    )
    return {
        "definition": definition_name,
        "local_run_dir": str(Path(local_results_dir) / definition_name),
    }


def _cli_prepare(args: argparse.Namespace) -> None:
    target = RemoteTarget(host=args.host, user=args.user, key_file=args.key_file)
    info = prepare_session(
        target, args.definition, args.dataset, args.author, args.baseline_author, args.isa,
        remote_root=args.remote_root, sync_repo=not args.no_sync,
        local_repo_dir=args.local_repo_dir, skip_preflight=args.skip_preflight,
        transport=args.transport,
    )
    if info["transport"] == "stdio":
        print("spawn command:")
        print(f"  {info['command']} {' '.join(info['args'])}")
    else:
        print(f"tunnel up: {info['endpoint']}")
        print("(Ctrl-C to tear down)")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            stop_tunnel(info)


def _cli_sync(args: argparse.Namespace) -> None:
    target = RemoteTarget(host=args.host, user=args.user, key_file=args.key_file)
    result = sync_results(
        target, args.definition, remote_root=args.remote_root,
        local_results_dir=args.local_results_dir,
    )
    print(result)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare-session")
    prep.add_argument("--host", required=True)
    prep.add_argument("--user", default="ubuntu")
    prep.add_argument("--key-file", default="~/.ssh/id_rsa")
    prep.add_argument("--remote-root", default="~/arm-bench")
    prep.add_argument("--dataset", required=True, choices=["ncnn", "simd-loop", "llama.cpp"])
    prep.add_argument("--definition", required=True)
    prep.add_argument("--author", default="nanobot")
    prep.add_argument("--baseline-author", required=True)
    prep.add_argument("--isa", required=True, choices=["neon", "sve", "sve2", "sme2"])
    prep.add_argument("--transport", choices=["stdio", "sse"], default="stdio")
    prep.add_argument("--local-repo-dir", help="Required unless --no-sync.")
    prep.add_argument("--no-sync", action="store_true")
    prep.add_argument("--skip-preflight", action="store_true",
                       help="Skip ensure_dataset_ready + ensure_baseline_collected "
                            "(use if you already know this instance is ready).")
    prep.set_defaults(func=_cli_prepare)

    sync = sub.add_parser("sync-results")
    sync.add_argument("--host", required=True)
    sync.add_argument("--user", default="ubuntu")
    sync.add_argument("--key-file", default="~/.ssh/id_rsa")
    sync.add_argument("--remote-root", default="~/arm-bench")
    sync.add_argument("--definition", required=True)
    sync.add_argument("--local-results-dir", required=True)
    sync.set_defaults(func=_cli_sync)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
