"""Minimal SSH/rsync helpers used only by mcp_app/smoke_test_driver.py.

Deliberately independent of skills/nanobot/nanobot-kernel-session/scripts/remote.py's
RemoteTarget — mcp_app and skills/ never import from each other (see
mcp_app/README.md). This is a handful of plain functions, not a reusable
class, because smoke_test_driver.py is a single linear loop, not something
that needs RemoteTarget's reusability.

Style reference only: eval/provision.py's InstanceHandle has the same shape
of ssh_cmd/rsync_to/rsync_from methods — never imported from here.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _ssh_base_args(key_file: str) -> list[str]:
    key = os.path.expanduser(key_file)
    return [
        "-i", key,
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=10",
        # The long-lived MCP session (ssh_spawn_args) can sit with zero
        # protocol traffic for many minutes at a time while evaluate()/
        # submit() crunch on the remote side — with no keepalive, a
        # connection silently dropped by some idle-TCP-timeout in the
        # network path (home router/NAT, not the AWS instance — sshd
        # restarts/OOM were ruled out separately) goes undetected for a
        # very long time (client just hangs) or forever. These two options
        # send an active probe every 15s and give up after 3 misses (45s),
        # both keeping the connection alive and detecting a truly dead one
        # quickly instead of hanging or waiting on OS-level TCP timeouts.
        "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=3",
    ]


def ssh_spawn_args(host: str, user: str, key_file: str, remote_cmd: str) -> list[str]:
    """Args for spawning `ssh` as a subprocess (e.g. as an MCP stdio client's command)."""
    return [*_ssh_base_args(key_file), f"{user}@{host}", remote_cmd]


def run_remote(
    host: str, user: str, key_file: str, remote_cmd: str, timeout: int = 120,
) -> tuple[int, str, str]:
    """Blocking SSH exec + capture. Used for the baseline-collection check script
    and per-definition `bench.cli collect-baselines` calls in driver.py."""
    result = subprocess.run(
        ["ssh", *ssh_spawn_args(host, user, key_file, remote_cmd)],
        capture_output=True, text=True, timeout=timeout,
    )
    return result.returncode, result.stdout, result.stderr


def rsync_to(
    host: str, user: str, key_file: str, local_dir: str | Path, remote_dir: str,
    paths: list[str],
) -> None:
    """Sync only `paths` (repo-root-relative files/dirs) from local_dir into
    remote_dir, one rsync per path with --delete. Allow-list, not a deny-list:
    anything not in `paths` (e.g. results/, agent-runs/, notebooks/ — generated
    directly on the remote by eval runs) is never touched by this call, so it
    can't accumulate stale synced copies that a deny-list would leave behind
    forever once excluded (rsync --exclude never deletes what's already there).
    """
    run_remote(host, user, key_file, f"mkdir -p {remote_dir}")
    ssh_opt = f"ssh {' '.join(_ssh_base_args(key_file))}"
    for rel in paths:
        local_path = Path(local_dir) / rel
        cmd = ["rsync", "-avz", "--delete", "-e", ssh_opt]
        if local_path.is_dir():
            cmd += [str(local_path) + "/", f"{user}@{host}:{remote_dir}/{rel}/"]
        else:
            cmd += [str(local_path), f"{user}@{host}:{remote_dir}/{rel}"]
        subprocess.run(cmd, check=True, capture_output=True)


def rsync_from(
    host: str, user: str, key_file: str, remote_path: str, local_dir: str | Path,
    excludes: list[str] | None = None,
) -> None:
    cmd = [
        "rsync", "-avz",
        "-e", f"ssh {' '.join(_ssh_base_args(key_file))}",
    ]
    for exc in (excludes or []):
        cmd += ["--exclude", exc]
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    cmd += [f"{user}@{host}:{remote_path}", str(local_dir) + "/"]
    subprocess.run(cmd, check=True, capture_output=True)


__all__ = ["ssh_spawn_args", "run_remote", "rsync_to", "rsync_from"]
