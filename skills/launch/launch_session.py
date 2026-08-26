"""launch_session — bring up one mcp_app session on a remote instance, sync
results back afterward. Runs on the caller's host, not the target instance.

Provisioning (`provision`/`teardown`/`status`) is done by the standalone
`eval/provision.py` script — invoked only via subprocess, never imported.
This module has zero Python imports from eval/ or mcp_app/ (see remote.py's
docstring); it only reads the shared `eval/eval_config.json` that
`eval/provision.py` writes, which is a file-format contract, not a Python
import. This module already assumes the full repo checkout (including
eval/) is present locally, since it rsyncs REPO_ROOT to the remote. Sharing
that one config file is what lets `eval/provision.py` and this module
provision/reuse/teardown the same instances without either side going stale
about what the other has done.
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
import http.client
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
# This module's own directory, so `from remote import RemoteTarget` below
# resolves whether this file is run as a script (already implicit — Python
# puts the script's own dir on sys.path[0]) or imported as a package
# submodule (e.g. `from skills.launch import launch_session`, which does
# NOT add this directory automatically — see eval/mcp_client.py).
sys.path.insert(0, str(Path(__file__).parent))

# contracts.py lives at the repo root, outside both eval/ and mcp_app/, so
# importing it doesn't violate this module's "zero imports from eval/ or
# mcp_app/" boundary (see module docstring).
from contracts import ISA_INSTANCE_MAP
from remote import RemoteTarget

EVAL_CONFIG_PATH = REPO_ROOT / "eval" / "eval_config.json"
PROVISION_SCRIPT = REPO_ROOT / "eval" / "provision.py"

# Repo-root-relative paths mcp_app/bench actually need on the remote side.
# Allow-list, not a deny-list — see RemoteTarget.rsync_to's docstring.
# TODO: fold into an env var (shared with the separately-duplicated copies in
# eval/provision.py and mcp_app/smoke_test_driver.py).
RSYNC_ALLOWLIST = ["bench", "bench-trace", "mcp_app", "requirements.txt"]

# Shared with eval/provision.py and mcp_app/smoke_test_driver.py — lives at
# the repo root (like contracts.py/config/kernel_contracts.yaml) so none of
# the three packages "owns" a separately-duplicated copy that can drift.
DATASET_BUILDS: dict = json.loads((REPO_ROOT / "config" / "dataset_builds.json").read_text())


@dataclass(frozen=True)
class ProvisionedInstance:
    target: RemoteTarget
    instance_type: str
    instance_id: Optional[str] = None


# label `dataset` here can be a single string
# or a list (`launch`'s repeatable --dataset, for dispatcher mode serving
# several datasets off one instance) — both fold into the same label.
def _dataset_label_part(dataset) -> str:
    if isinstance(dataset, list):
        return "-".join(sorted(dataset))
    return dataset or ""


# label = f"{dataset(s)}-{author}" 
def _label_for(dataset, author: str) -> str:
    raw = f"{_dataset_label_part(dataset)}-{author}"
    return re.sub(r"[^a-z0-9.-]", "-", raw.lower()).strip("-.")


def _read_config_instance(label: str) -> Optional[ProvisionedInstance]:
    """Read the shared eval/eval_config.json directly for a running instance
    under `label`. No import from eval/provision.py."""
    if not EVAL_CONFIG_PATH.exists():
        return None
    config = json.loads(EVAL_CONFIG_PATH.read_text())
    inst = config.get("instances", {}).get(label, {})
    host = inst.get("host", "")
    if not host:
        return None
    return ProvisionedInstance(
        target=RemoteTarget(host=host, user=inst.get("user", "ubuntu"),
                             key_file=inst.get("key_file", "~/.ssh/id_rsa")),
        instance_type=inst.get("instance_type", "c7g.large"),
        instance_id=inst.get("instance_id"),
    )


def _provision(
    isa: str, instance_type: str, dataset: str, *, label: str, on_demand: bool = False
) -> ProvisionedInstance:
    """Subprocess-invoke the standalone eval/provision.py, then read the
    eval_config.json it wrote. Reuses a reachable instance under `label`
    if one's already up; otherwise provisions a fresh one.

    `on_demand` only takes effect when running a long-run job."""
    cmd = [sys.executable, str(PROVISION_SCRIPT),
           "--isa", isa, "--instance", instance_type, "--dataset", dataset,
           "--label", label]
    if on_demand:
        cmd.append("--on-demand")
    subprocess.run(cmd, check=True)
    instance = _read_config_instance(label)
    if instance is None:
        raise RuntimeError(
            f"eval/provision.py exited successfully but wrote no instance for label={label!r}"
        )
    return instance


def _teardown(label: Optional[str] = None) -> None:
    """`label` given: destroy just that one instance. Omitted: tear down
    every label eval/provision.py knows about (old "destroy everything"
    behavior, kept as an explicit opt-in — see its own teardown() docstring).
    """
    cmd = [sys.executable, str(PROVISION_SCRIPT), "--teardown"]
    if label:
        cmd += ["--label", label]
    subprocess.run(cmd, check=True)


def _status() -> None:
    config = json.loads(EVAL_CONFIG_PATH.read_text()) if EVAL_CONFIG_PATH.exists() else {}
    if not config.get("instances"):
        print("No eval/eval_config.json instances found. Run `provision` first.")
        return
    for label, inst in config["instances"].items():
        host = inst.get("host", "")
        instance_type = inst.get("instance_type", "?")
        print(f"  {label} ({instance_type}): {host or 'not provisioned'}")


def _spawn_command(
    target: RemoteTarget, remote_root: str, datasets: list[str],
    author: str, baseline_author: Optional[str], isa: str, *, port: int,
) -> str:
    """Remote command for a persistent streamable-http-mode mcp_app.server
    (see prepare_session's docstring for why this is the only mode this
    script offers — mcp_app/smoke_test_driver.py still uses stdio directly,
    this is unrelated to that). `datasets` with more than one entry starts
    mcp_app.server's dispatcher mode (repeated --dataset flags) — see
    mcp_app/agent_tools/dispatcher.py."""
    run_dir = f"{remote_root}/agent-runs-mcp/{author}"
    dataset_flags = " ".join(f"--dataset {ds}" for ds in datasets)
    cmd = (
        f"cd {remote_root} && python3 -m mcp_app.server {dataset_flags} "
        f"--author {author} --isa {isa} --run-dir {run_dir} "
        f"--transport streamable-http --bind-host 127.0.0.1 --port {port}"
    )
    if baseline_author:
        cmd += f" --baseline-author {baseline_author}"
    return cmd


# ── Pre-flight: make sure the instance actually has what a session needs
#    before nanobot ever connects. mcp_app/smoke_test_driver.py (mcp_app's
#    non-nanobot smoke-test tool) has its own independent copy of this same
#    logic, since mcp_app and skills/ never import from each other.

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
    dataset: str | list[str],
    author: str,
    isa: str,
    *,
    baseline_author: Optional[str] = None,
    remote_root: str = "~/arm-bench",
    sync_repo: bool = True,
    local_repo_dir: Optional[str | Path] = None,
    local_port: Optional[int] = None,
    remote_port: int = 8765,
    startup_timeout: int = 60,
) -> dict:
    """Get an mcp_app session ready to be driven by a real MCP client.

    `dataset` accepts either a single dataset string (today's behavior,
    unchanged) or a list of more than one — the remote mcp_app.server then
    starts in dispatcher mode, serving all of them over one connection (see
    mcp_app/agent_tools/dispatcher.py). `baseline_author` is a single-dataset
    override only — omit it and the server auto-derives it from `dataset`
    (mcp_app/agent_tools/baseline_readiness.py::DEFAULT_BASELINE_AUTHOR); it
    can't be combined with more than one dataset, since one override can't
    correctly apply to more than one dataset's baseline.

    Always use streamable-http: establishes an SSH local-port-forward +
    starts the remote server, returns {"transport": "streamable-http",
    "endpoint": "http://127.0.0.1:<port>/mcp", "_tunnel_proc": <Popen>} —
    call stop_tunnel() on the result when done. The SSH tunnel (not the
    server's own transport) is what keeps the compile/evaluate tool surface
    off the public network — see mcp_app/server.py's module docstring.
    """
    datasets = [dataset] if isinstance(dataset, str) else list(dict.fromkeys(dataset))
    if len(datasets) > 1 and baseline_author is not None:
        raise ValueError(
            "baseline_author can't be used with more than one dataset — each "
            "dataset auto-derives its own (see DEFAULT_BASELINE_AUTHOR)."
        )

    if sync_repo:
        if local_repo_dir is None:
            raise ValueError("local_repo_dir is required when sync_repo=True")
        target.rsync_to(local_repo_dir, remote_root, paths=RSYNC_ALLOWLIST)

    for ds in datasets:
        ensure_dataset_ready(target, ds)

    remote_cmd = _spawn_command(
        target, remote_root, datasets, author, baseline_author, isa,
        port=remote_port,
    )
    ssh_cmd = [
        "ssh", "-L", f"{local_port}:127.0.0.1:{remote_port}",
        *target.ssh_base_args(), f"{target.user}@{target.host}", remote_cmd,
    ]
    proc = subprocess.Popen(ssh_cmd)
    endpoint = f"http://127.0.0.1:{local_port}/mcp"
    try:
        _wait_for_port(local_port, timeout=startup_timeout, proc=proc)
    except BaseException:
        # Any exception will kill the listening local port, including ctrl+C
        print(f"[launch_session] kill listening local port...")
        proc.kill()
        proc.wait()
        _kill_remote_port(target, remote_port)
        print(f"[launch_session] listening local port successfully killed")
        raise
    return {
        "transport": "streamable-http", "endpoint": endpoint, "_tunnel_proc": proc,
        "_target": target, "_remote_port": remote_port,
    }


def _kill_remote_port(target: RemoteTarget, remote_port: int) -> None:
    """Explicitly kill whatever's bound to remote_port on target, synchronously.
    """
    target.run(
        f"fuser -k {remote_port}/tcp 2>/dev/null || "
        f"pkill -f 'mcp_app.server.*--port {remote_port}' 2>/dev/null || true",
        timeout=15,
    )


def _probe_ready(port: int) -> bool:
    """One lightweight HTTP round-trip against the streamable-http endpoint.
    verify if the real mcp server can answer request
    """
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    try:
        conn.request("POST", "/mcp", body=b"{}", headers={"Content-Type": "application/json"})
        conn.getresponse()
        return True
    except (OSError, http.client.HTTPException):
        return False
    finally:
        conn.close()


def _wait_for_port(port: int, *, timeout: float, proc: subprocess.Popen) -> None:
    """Wait until the remote mcp_app.server is actually answering requests
    through the tunnel — not just until ssh's local listener is up (see
    _probe_ready's docstring for why that distinction matters)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"ssh tunnel process exited early (rc={proc.returncode})")
        if _probe_ready(port):
            return
        time.sleep(0.3)
    raise TimeoutError(f"mcp_app.server on 127.0.0.1:{port} not answering after {timeout}s")


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

    target: Optional[RemoteTarget] = prepared.get("_target")
    remote_port = prepared.get("_remote_port")
    if target is not None and remote_port is not None:
        _kill_remote_port(target, remote_port)


def sync_results(
    target: RemoteTarget,
    author: str,
    *,
    definition: Optional[str] = None,
    remote_root: str = "~/arm-bench",
    local_results_dir: str | Path,
    sync_bench_trace: bool = False,
    local_bench_trace_dir: Optional[str | Path] = None,
) -> dict:
    """Pull this author's session results back to local_results_dir.

    Pulls the whole `agent-runs-mcp/<author>/` directory (every definition
    that author's session touched) unless `definition` is given, in which
    case only that one definition's subdirectory is synced.

    `sync_bench_trace=True` additionally pulls back `bench-trace/solutions/`
    and `bench-trace/traces/` from the remote instance 
    """
    remote_dir = f"agent-runs-mcp/{author}"
    if definition:
        remote_dir += f"/{definition}"
    target.rsync_from(f"{remote_root}/{remote_dir}", local_results_dir)
    result = {
        "author": author,
        "definition": definition,
        "local_run_dir": str(Path(local_results_dir) / Path(remote_dir).name),
    }
    if sync_bench_trace:
        bt_dir = Path(local_bench_trace_dir) if local_bench_trace_dir else REPO_ROOT / "bench-trace"
        # sync back new solutions and traces for kernel stability test
        target.rsync_from(f"{remote_root}/bench-trace/solutions/", bt_dir / "solutions")
        target.rsync_from(f"{remote_root}/bench-trace/traces/", bt_dir / "traces")
        result["local_bench_trace_dir"] = str(bt_dir)
    return result


def _cli_prepare(args: argparse.Namespace) -> None:
    target = RemoteTarget(host=args.host, user=args.user, key_file=args.key_file)
    info = prepare_session(
        target, args.dataset, args.author, args.isa,
        baseline_author=args.baseline_author,
        remote_root=args.remote_root, sync_repo=not args.no_sync,
        local_repo_dir=args.local_repo_dir,
        local_port=args.local_port, remote_port=args.remote_port,
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
        sync_bench_trace=args.sync_bench_trace,
    )
    print(result)


def _resolve_instance(args: argparse.Namespace) -> ProvisionedInstance:
    """Reuse an already-up-and-reachable instance for --isa if one's up,
    otherwise provision a fresh one — via eval/provision.py's own
    reuse-if-reachable default (see its module docstring). Note:
    eval/provision.py always rsyncs its own repo checkout during
    provisioning, so `--local-repo-dir` has no effect on that initial sync;
    `_cli_launch` re-syncs via `prepare_session()` afterward, which does
    respect it.

    eval/provision.py's own `--dataset` only builds one dataset's native lib
    at provision time. With more than one --dataset requested here, skip
    that step (pass "") and rely on prepare_session's own per-dataset
    ensure_dataset_ready loop right after — slower on a cold instance's very
    first multi-dataset launch, correct thereafter, no eval/ changes needed.
    """
    instance_type = args.instance or ISA_INSTANCE_MAP.get(args.isa, "c7g.large")
    provision_dataset = args.dataset[0] if len(args.dataset) == 1 else ""
    author = getattr(args, "author", None)
    default_label = (
        _label_for(args.dataset, author) if author is not None
        else f"{_dataset_label_part(args.dataset)}-{args.isa}"
    )
    label = args.label or default_label
    return _provision(args.isa, instance_type, provision_dataset, label=label,
                       on_demand=args.on_demand)


def _cli_provision(args: argparse.Namespace) -> None:
    instance = _resolve_instance(args)
    t = instance.target
    print(f"host={t.host} user={t.user} key_file={t.key_file} instance_type={instance.instance_type}")


def _cli_teardown(args: argparse.Namespace) -> None:
    _teardown(args.label)


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
        local_port=args.local_port, remote_port=args.remote_port,
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
    prep.add_argument("--dataset", action="append", required=True,
                       choices=["ncnn", "simd-loop", "llama.cpp"],
                       help="Repeatable — pass more than once to start the remote "
                            "mcp_app.server in dispatcher mode, serving several "
                            "datasets over one connection (see "
                            "mcp_app/agent_tools/dispatcher.py). A single --dataset "
                            "behaves exactly as before.")
    prep.add_argument("--author", default="nanobot")
    prep.add_argument("--baseline-author", default=None,
                       help="Override only — the server auto-derives this from --dataset.")
    prep.add_argument("--isa", required=True, choices=["neon", "sve", "sve2", "sme2"])
    prep.add_argument("--local-port", type=int, default=8888,
                       help="Fix the local tunnel port instead of picking a random free "
                            "one each run, so a reused mcp client config (e.g. nanobot's) "
                            "doesn't need editing every relaunch.")
    prep.add_argument("--remote-port", type=int, default=8765,
                       help="Port mcp_app.server binds to on the remote instance. Override "
                            "when reusing the same instance (same --isa) for more than one "
                            "concurrent session, so the servers don't collide on 8765.")
    prep.add_argument("--local-repo-dir", help="Required unless --no-sync.")
    prep.add_argument("--no-sync", action="store_true")
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
    sync.add_argument("--sync-bench-trace", action="store_true",
                       help="Also pull back bench-trace/solutions/ and bench-trace/traces/ "
                            "from the remote instance (merge-pull, no --delete) ")
    sync.set_defaults(func=_cli_sync)

    def _add_provision_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--isa", required=True, choices=["neon", "sve", "sve2", "sme2"])
        sp.add_argument("--label", default=None,
                         help="Name identifying this instance — one per concurrently-desired "
                              "instance (see eval/provision.py's module docstring). Default: "
                              "f'{dataset(s)}-{author}' for `launch` (which has --author), "
                              "f'{dataset(s)}-{isa}' for standalone `provision`.")
        sp.add_argument("--instance", default=None,
                         help="EC2 instance type override (e.g. c8g.xlarge). "
                              "Defaults to ISA_INSTANCE_MAP[isa].")
        sp.add_argument("--local-repo-dir", default=None,
                         help="Repo checkout for prepare_session's rsync (the `launch` "
                              "subcommand only — eval/provision.py always rsyncs its own "
                              "repo root during provisioning itself). Defaults to this "
                              "repo's own root.")
        sp.add_argument("--on-demand", action="store_true",
                         help="Provision on-demand instead of the default spot — AWS won't "
                              "reclaim the instance mid-run (e.g. during a long unattended "
                              "fleet run), at a higher hourly price. Only takes effect when "
                              "actually provisioning a fresh instance; an existing reachable "
                              "one is reused as-is regardless — run the `teardown` subcommand "
                              "first to force a clean on-demand replacement.")

    prov = sub.add_parser("provision", help="Bring up (or reuse) a Graviton instance for --isa.")
    _add_provision_args(prov)
    prov.add_argument("--dataset", default="", choices=["", "ncnn", "simd-loop", "llama.cpp"],
                       help="Build this dataset's native lib after provisioning. Empty = skip.")
    prov.set_defaults(func=_cli_provision)

    teardown_p = sub.add_parser("teardown", help="Terraform-destroy the instance(s).")
    teardown_p.add_argument("--label", default=None,
                             help="Destroy just this one instance. Omit to tear down every "
                                  "label eval/eval_config.json knows about.")
    teardown_p.set_defaults(func=_cli_teardown)

    status_p = sub.add_parser("status", help="Show eval/eval_config.json's tracked instances.")
    status_p.set_defaults(func=_cli_status)

    launch = sub.add_parser(
        "launch",
        help="provision (or reuse) an instance for --isa, then start an mcp_app session on it.",
    )
    _add_provision_args(launch)
    launch.add_argument("--dataset", action="append", required=True,
                         choices=["ncnn", "simd-loop", "llama.cpp"],
                         help="Repeatable — pass more than once to start the remote "
                              "mcp_app.server in dispatcher mode, serving several "
                              "datasets over one connection (see "
                              "mcp_app/agent_tools/dispatcher.py). A single --dataset "
                              "behaves exactly as before.")
    launch.add_argument("--author", default="nanobot")
    launch.add_argument("--baseline-author", default=None,
                         help="Override only — the server auto-derives this from --dataset.")
    launch.add_argument("--remote-root", default="~/arm-bench")
    launch.add_argument("--local-port", type=int, default=8888,
                         help="Fix the local tunnel port instead of picking a random free "
                              "one each run, so a reused mcp client config (e.g. nanobot's) "
                              "doesn't need editing every relaunch.")
    launch.add_argument("--remote-port", type=int, default=8765,
                         help="Port mcp_app.server binds to on the remote instance. Override "
                              "when reusing the same instance (same --isa) for more than one "
                              "concurrent session, so the servers don't collide on 8765.")
    launch.add_argument("--no-sync", action="store_true",
                         help="Skip prepare_session's own rsync (provision already synced once).")
    launch.set_defaults(func=_cli_launch)

    args = p.parse_args(argv)
    if isinstance(getattr(args, "dataset", None), list):
        args.dataset = list(dict.fromkeys(args.dataset))  # dedupe, preserve order
    args.func(args)


if __name__ == "__main__":
    main()
