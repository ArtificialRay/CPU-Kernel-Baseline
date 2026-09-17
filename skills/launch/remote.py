"""RemoteTarget — self-contained SSH/rsync helper for this skill's own scripts.

Zero imports from mcp_app or eval/ — this skill lives at the repo root as a
sibling to mcp_app/, not nested inside it, and mcp_app/smoke_test_driver.py
has its own independent equivalent (mcp_app/scripts/_local_ssh.py) rather
than importing this file.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RemoteTarget:
    host: str
    user: str
    key_file: str

    def ssh_base_args(self) -> list[str]:
        key = os.path.expanduser(self.key_file)
        return [
            "-i", key,
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=10",
            # The long-lived MCP session (ssh_spawn_args) can sit with zero
            # protocol traffic for many minutes at a time while evaluate()/
            # submit() crunch on the remote side
            "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=3"
        ]

    def ssh_cmd(self, remote_cmd: str) -> list[str]:
        return ["ssh", *self.ssh_base_args(), f"{self.user}@{self.host}", remote_cmd]

    def run(self, remote_cmd: str, timeout: int = 120) -> tuple[int, str, str]:
        result = subprocess.run(
            self.ssh_cmd(remote_cmd), capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr

    def rsync_to(
        self, local_dir: str | Path, remote_dir: str, paths: list[str],
    ) -> None:
        """Sync only `paths` (repo-root-relative files/dirs) from local_dir into
        remote_dir, one rsync per path with --delete. Allow-list, not a deny-list:
        anything not in `paths` (e.g. results/, agent-runs/, notebooks/ — generated
        directly on the remote by eval runs) is never touched by this call, so it
        can't accumulate stale synced copies that a deny-list would leave behind
        forever once excluded (rsync --exclude never deletes what's already there).
        """
        # Create each path's parent too: entries may be nested (e.g.
        # "bench-trace/definitions"), and rsync does not create missing
        # intermediate directories — it fails with
        # `mkdir ... failed: No such file or directory`.
        # Unquoted on purpose: remote_dir is a repo constant like "~/arm-bench"
        # and the tilde must stay shell-expandable (quoting it would create a
        # directory literally named "~").
        parents = sorted({f"{remote_dir}/{rel}".rsplit("/", 1)[0] for rel in paths})
        self.run("mkdir -p " + " ".join(dict.fromkeys([remote_dir, *parents])))
        ssh_opt = f"ssh {' '.join(self.ssh_base_args())}"
        for rel in paths:
            local_path = Path(local_dir) / rel
            cmd = ["rsync", "-avz", "--delete", "-e", ssh_opt]
            if local_path.is_dir():
                cmd += [str(local_path) + "/", f"{self.user}@{self.host}:{remote_dir}/{rel}/"]
            else:
                cmd += [str(local_path), f"{self.user}@{self.host}:{remote_dir}/{rel}"]
            subprocess.run(cmd, check=True, capture_output=True)

    def rsync_from(
        self, remote_path: str, local_dir: str | Path, excludes: list[str] | None = None,
    ) -> None:
        cmd = ["rsync", "-avz", "-e", f"ssh {' '.join(self.ssh_base_args())}"]
        for exc in (excludes or []):
            cmd += ["--exclude", exc]
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        cmd += [f"{self.user}@{self.host}:{remote_path}", str(local_dir) + "/"]
        subprocess.run(cmd, check=True, capture_output=True)


__all__ = ["RemoteTarget"]
