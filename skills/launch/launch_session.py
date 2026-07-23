"""launch_session — bring up one mcp_app session on a remote instance, sync
results back afterward. Runs on the caller's host, not the target instance.

Provisioning (`provision`/`teardown`/`status`) is done by the standalone
`eval/provision.py` script — invoked only via subprocess, never imported.
This module has zero Python imports from eval/ or mcp_app/ (see remote.py's
docstring and mcp_app/README.md's "Scope boundary" section); it only reads
the shared `eval/eval_config.json` that `eval/provision.py` writes, which is
a file-format contract, not a Python import. This module already assumes
the full repo checkout (including eval/) is present locally, since it
rsyncs REPO_ROOT to the remote. Sharing that one config file (instead of
this skill keeping its own, as it used to) is what lets `eval/provision.py`
and this module provision/reuse/teardown the same instances without either
side going stale about what the other has done.
`launch` composes provisioning + `prepare_session()` in one call.
`prepare-session`/`sync-results` stay separate commands: `prepare-session`
blocks in the foreground for as long as you want the tunnel + remote server
alive (Ctrl-C tears it down — see `stop_tunnel()`), while an MCP client
drives the actual optimization session against it from a separate process;
`sync-results` is meant to run afterward, once that session is done, to
pull results back — not something this script's own lifecycle could know
the right moment for on its own.
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from remote import RemoteTarget

REPO_ROOT = Path(__file__).parent.parent.parent
EVAL_CONFIG_PATH = REPO_ROOT / "eval" / "eval_config.json"
PROVISION_SCRIPT = REPO_ROOT / "eval" / "provision.py"

# Own copy of eval/provision.py's ISA_INSTANCE_MAP — small (4 rows), rarely
# changes, kept in sync by hand rather than generated (same tradeoff as the
# dataset/baseline_author/isa table in SKILL.md).
ISA_INSTANCE_MAP = {
    "neon": "c7g.large",
    "sve": "c7g.large",
    "sve2": "c8g.large",
    "sme2": "c8g.large",
}

# Repo-root-relative paths mcp_app/bench actually need on the remote side.
# Allow-list, not a deny-list — see RemoteTarget.rsync_to's docstring.
# TODO: fold into an env var (shared with the separately-duplicated copies in
# eval/provision.py and mcp_app/smoke_test_driver.py).
RSYNC_ALLOWLIST = ["bench", "bench-trace", "mcp_app", "requirements.txt"]

# Directory to this skill's own copy of eval/dataset_builds.json's content
DATASET_BUILDS: dict = json.loads((Path(__file__).parent / "dataset_builds.json").read_text())


@dataclass(frozen=True)
class ProvisionedInstance:
    target: RemoteTarget
    instance_type: str
    instance_id: Optional[str] = None


def _tier_for_isa(isa: str) -> str:
    return "c8g" if isa in ("sve2", "sme2") else "c7g"


def _read_config_instance(isa: str) -> Optional[ProvisionedInstance]:
    """Read the shared eval/eval_config.json directly for a running instance
    of `isa`'s tier. No import from eval/provision.py."""
    if not EVAL_CONFIG_PATH.exists():
        return None
    config = json.loads(EVAL_CONFIG_PATH.read_text())
    tier = _tier_for_isa(isa)
    inst = config.get("instances", {}).get(tier, {})
    host = inst.get("host", "")
    if not host:
        return None
    return ProvisionedInstance(
        target=RemoteTarget(host=host, user=inst.get("user", "ubuntu"),
                             key_file=inst.get("key_file", "~/.ssh/id_rsa")),
        instance_type=inst.get("instance_type", ISA_INSTANCE_MAP.get(isa, "c7g.large")),
        instance_id=inst.get("instance_id"),
    )


def _provision(isa: str, instance_type: str, dataset: str) -> ProvisionedInstance:
    """Subprocess-invoke the standalone eval/provision.py, then read the
    eval_config.json it wrote. Reuses a reachable instance for this ISA
    tier if one's already up; otherwise provisions a fresh one."""
    subprocess.run(
        [sys.executable, str(PROVISION_SCRIPT),
         "--isa", isa, "--instance", instance_type, "--dataset", dataset],
        check=True,
    )
    instance = _read_config_instance(isa)
    if instance is None:
        raise RuntimeError(
            f"eval/provision.py exited successfully but wrote no instance for isa={isa!r}"
        )
    return instance


def _teardown() -> None:
    subprocess.run([sys.executable, str(PROVISION_SCRIPT), "--teardown"], check=True)


def _status() -> None:
    config = json.loads(EVAL_CONFIG_PATH.read_text()) if EVAL_CONFIG_PATH.exists() else {}
    if not config.get("instances"):
        print("No eval/eval_config.json instances found. Run `provision` first.")
        return
    for tier, inst in config["instances"].items():
        host = inst.get("host", "")
        print(f"  {tier}: {host or 'not provisioned'}")


def _spawn_command(
    target: RemoteTarget, remote_root: str, dataset: str,
    author: str, baseline_author: Optional[str], isa: str, *, port: int,
) -> str:
    """Remote command for a persistent sse-mode mcp_app.server (see
    prepare_session's docstring for why this is the only mode this script
    offers — mcp_app/smoke_test_driver.py still uses stdio directly, this
    is unrelated to that)."""
    run_dir = f"{remote_root}/agent-runs-mcp/{author}"
    cmd = (
        f"cd {remote_root} && python3 -m mcp_app.server --dataset {dataset} "
        f"--author {author} --isa {isa} --run-dir {run_dir} "
        f"--transport sse --bind-host 127.0.0.1 --port {port}"
    )
    if baseline_author:
        cmd += f" --baseline-author {baseline_author}"
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


def prepare_session(
    target: RemoteTarget,
    dataset: str,
    author: str,
    isa: str,
    *,
    baseline_author: Optional[str] = None,
    remote_root: str = "~/arm-bench",
    sync_repo: bool = True,
    local_repo_dir: Optional[str | Path] = None,
    skip_preflight: bool = False,
    local_port: Optional[int] = None,
    remote_port: int = 8765,
    startup_timeout: int = 60,
) -> dict:
    """Get an mcp_app session ready to be driven by a real MCP client.

    `baseline_author` is an override only — omit it and the server
    auto-derives it from `dataset`
    (mcp_app/agent_tools/baseline_readiness.py::DEFAULT_BASELINE_AUTHOR).

    Always use sse: establishes an SSH local-port-forward + starts the remote
    server, returns {"transport": "sse", "endpoint": "http://127.0.0.1:<port>",
    "_tunnel_proc": <Popen>} — call stop_tunnel() on the result when done.
    """
    if sync_repo:
        if local_repo_dir is None:
            raise ValueError("local_repo_dir is required when sync_repo=True")
        target.rsync_to(local_repo_dir, remote_root, paths=RSYNC_ALLOWLIST)

    if not skip_preflight:
        ensure_dataset_ready(target, dataset)

    if local_port is None:
        local_port = _free_local_port()
    remote_cmd = _spawn_command(
        target, remote_root, dataset, author, baseline_author, isa,
        port=remote_port,
    )
    ssh_cmd = [
        "ssh", "-L", f"{local_port}:127.0.0.1:{remote_port}",
        *target.ssh_base_args(), f"{target.user}@{target.host}", remote_cmd,
    ]
    proc = subprocess.Popen(ssh_cmd)
    endpoint = f"http://127.0.0.1:{local_port}/sse"
    try:
        _wait_for_port(local_port, timeout=startup_timeout, proc=proc)
    except BaseException:
        # Any exception will kill the listening local port, including ctrl+C
        proc.kill()
        proc.wait()
        raise
    return {"transport": "sse", "endpoint": endpoint, "_tunnel_proc": proc}


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
    author: str,
    *,
    definition: Optional[str] = None,
    remote_root: str = "~/arm-bench",
    local_results_dir: str | Path,
) -> dict:
    """Pull this author's session results back to local_results_dir.

    Pulls the whole `agent-runs-mcp/<author>/` directory (every definition
    that author's session touched) unless `definition` is given, in which
    case only that one definition's subdirectory is synced.
    """
    remote_dir = f"agent-runs-mcp/{author}"
    if definition:
        remote_dir += f"/{definition}"
    target.rsync_from(f"{remote_root}/{remote_dir}", local_results_dir)
    return {
        "author": author,
        "definition": definition,
        "local_run_dir": str(Path(local_results_dir) / Path(remote_dir).name),
    }


def _cli_prepare(args: argparse.Namespace) -> None:
    target = RemoteTarget(host=args.host, user=args.user, key_file=args.key_file)
    info = prepare_session(
        target, args.dataset, args.author, args.isa,
        baseline_author=args.baseline_author,
        remote_root=args.remote_root, sync_repo=not args.no_sync,
        local_repo_dir=args.local_repo_dir, skip_preflight=args.skip_preflight,
        local_port=args.local_port,
    )
    try:
        print(f"tunnel up: {info['endpoint']}")
        print("(Ctrl-C to tear down)")
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        stop_tunnel(info)


def _cli_sync(args: argparse.Namespace) -> None:
    target = RemoteTarget(host=args.host, user=args.user, key_file=args.key_file)
    result = sync_results(
        target, args.author, definition=args.definition,
        remote_root=args.remote_root, local_results_dir=args.local_results_dir,
    )
    print(result)


def _resolve_instance(args: argparse.Namespace) -> ProvisionedInstance:
    """Reuse an already-up-and-reachable instance for --isa if one's up,
    otherwise provision a fresh one — via eval/provision.py's own
    reuse-if-reachable default (see its module docstring). Note:
    eval/provision.py always rsyncs its own repo checkout during
    provisioning, so `--local-repo-dir` has no effect on that initial sync;
    `_cli_launch` re-syncs via `prepare_session()` afterward, which does
    respect it."""
    instance_type = args.instance or ISA_INSTANCE_MAP.get(args.isa, "c7g.large")
    return _provision(args.isa, instance_type, args.dataset)


def _cli_provision(args: argparse.Namespace) -> None:
    instance = _resolve_instance(args)
    t = instance.target
    print(f"host={t.host} user={t.user} key_file={t.key_file} instance_type={instance.instance_type}")


def _cli_teardown(args: argparse.Namespace) -> None:
    _teardown()


def _cli_status(args: argparse.Namespace) -> None:
    _status()


def _cli_launch(args: argparse.Namespace) -> None:
    """Provision (or reuse) an instance for --isa, then start an mcp_app
    session on it — provisioning + `prepare_session` in one call. Always
    re-syncs the repo via prepare_session (cheap, delta-only) so a reused
    instance can't silently run stale code."""
    instance = _resolve_instance(args)
    target = instance.target
    info = prepare_session(
        target, args.dataset, args.author, args.isa,
        baseline_author=args.baseline_author,
        remote_root=args.remote_root, sync_repo=not args.no_sync,
        local_repo_dir=args.local_repo_dir or str(REPO_ROOT),
        skip_preflight=args.skip_preflight,
        local_port=args.local_port,
    )
    try:
        print(f"tunnel up: {info['endpoint']}")
        print("(Ctrl-C to tear down)")
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        stop_tunnel(info)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare-session")
    prep.add_argument("--host", required=True)
    prep.add_argument("--user", default="ubuntu")
    prep.add_argument("--key-file", default="~/.ssh/id_rsa")
    prep.add_argument("--remote-root", default="~/arm-bench")
    prep.add_argument("--dataset", required=True, choices=["ncnn", "simd-loop", "llama.cpp"])
    prep.add_argument("--author", default="nanobot")
    prep.add_argument("--baseline-author", default=None,
                       help="Override only — the server auto-derives this from --dataset.")
    prep.add_argument("--isa", required=True, choices=["neon", "sve", "sve2", "sme2"])
    prep.add_argument("--local-port", type=int, default=None,
                       help="Fix the local tunnel port instead of picking a random free "
                            "one each run, so a reused mcp client config (e.g. nanobot's) "
                            "doesn't need editing every relaunch.")
    prep.add_argument("--local-repo-dir", help="Required unless --no-sync.")
    prep.add_argument("--no-sync", action="store_true")
    prep.add_argument("--skip-preflight", action="store_true",
                       help="Skip ensure_dataset_ready (use if you already know this "
                            "instance's native-library build is ready).")
    prep.set_defaults(func=_cli_prepare)

    sync = sub.add_parser("sync-results")
    sync.add_argument("--host", required=True)
    sync.add_argument("--user", default="ubuntu")
    sync.add_argument("--key-file", default="~/.ssh/id_rsa")
    sync.add_argument("--remote-root", default="~/arm-bench")
    sync.add_argument("--author", default="nanobot")
    sync.add_argument("--definition", default=None,
                       help="Sync only this definition's subdirectory. Omit to sync everything "
                            "this author's session touched.")
    sync.add_argument("--local-results-dir", required=True)
    sync.set_defaults(func=_cli_sync)

    def _add_provision_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--isa", required=True, choices=["neon", "sve", "sve2", "sme2"])
        sp.add_argument("--instance", default=None,
                         help="EC2 instance type override (e.g. c8g.xlarge). "
                              "Defaults to ISA_INSTANCE_MAP[isa].")
        sp.add_argument("--local-repo-dir", default=None,
                         help="Repo checkout for prepare_session's rsync (the `launch` "
                              "subcommand only — eval/provision.py always rsyncs its own "
                              "repo root during provisioning itself). Defaults to this "
                              "repo's own root.")

    prov = sub.add_parser("provision", help="Bring up (or reuse) a Graviton instance for --isa.")
    _add_provision_args(prov)
    prov.add_argument("--dataset", default="", choices=["", "ncnn", "simd-loop", "llama.cpp"],
                       help="Build this dataset's native lib after provisioning. Empty = skip.")
    prov.set_defaults(func=_cli_provision)

    teardown_p = sub.add_parser("teardown", help="Terraform-destroy the instance(s).")
    teardown_p.set_defaults(func=_cli_teardown)

    status_p = sub.add_parser("status", help="Show eval/eval_config.json's tracked instances.")
    status_p.set_defaults(func=_cli_status)

    launch = sub.add_parser(
        "launch",
        help="provision (or reuse) an instance for --isa, then start an mcp_app session on it.",
    )
    _add_provision_args(launch)
    launch.add_argument("--dataset", required=True, choices=["ncnn", "simd-loop", "llama.cpp"])
    launch.add_argument("--author", default="nanobot")
    launch.add_argument("--baseline-author", default=None,
                         help="Override only — the server auto-derives this from --dataset.")
    launch.add_argument("--remote-root", default="~/arm-bench")
    launch.add_argument("--local-port", type=int, default=None,
                         help="Fix the local tunnel port instead of picking a random free "
                              "one each run, so a reused mcp client config (e.g. nanobot's) "
                              "doesn't need editing every relaunch.")
    launch.add_argument("--no-sync", action="store_true",
                         help="Skip prepare_session's own rsync (provision already synced once).")
    launch.add_argument("--skip-preflight", action="store_true",
                         help="Skip ensure_dataset_ready (use if you already know this "
                              "instance's native-library build is ready).")
    launch.set_defaults(func=_cli_launch)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
