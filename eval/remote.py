"""InstanceHandle — self-contained SSH/rsync helper for eval/'s own scripts.

Pure SSH client: given host/user/key_file for an instance that's already
up, run commands and move files on it. No provisioning/Terraform logic and
no config-file I/O lives here — that's eval/provision.py's job. Consumers
that just need to talk to an already-provisioned instance (evaluator.py,
agent_tools/base.py, run_benchmark.py) import from here, never from
eval/provision.py.

Style reference: skills/launch/remote.py's RemoteTarget has the same shape
of ssh_cmd/rsync_to/rsync_from methods (separately duplicated there, not
imported, per that skill's own zero-imports-from-eval/ boundary).
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class InstanceHandle:
    host: str
    user: str
    key_file: str
    instance_type: str
    instance_id: str | None = None

    def ssh_base_args(self) -> list[str]:
        key = os.path.expanduser(self.key_file)
        return [
            "-i", key,
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=10",
        ]

    def ssh_cmd(self, remote_cmd: str) -> list[str]:
        return ["ssh"] + self.ssh_base_args() + [f"{self.user}@{self.host}", remote_cmd]

    def run(self, remote_cmd: str, timeout: int = 120) -> tuple[int, str, str]:
        result = subprocess.run(
            self.ssh_cmd(remote_cmd),
            capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr

    def upload_file(self, local_path: str, remote_path: str):
        key = os.path.expanduser(self.key_file)
        subprocess.run([
            "scp",
            "-i", key,
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            str(local_path),
            f"{self.user}@{self.host}:{remote_path}",
        ], check=True, capture_output=True)

    def rsync_to(self, local_dir: str, remote_dir: str, paths: list[str]):
        """Sync only `paths` (repo-root-relative files/dirs) from local_dir into
        remote_dir, one rsync per path with --delete. Allow-list, not a deny-list:
        anything not in `paths` (e.g. results/, agent-runs/, notebooks/ — generated
        directly on the remote by eval runs) is never touched by this call, so it
        can't accumulate stale synced copies that a deny-list would leave behind
        forever once excluded (rsync --exclude never deletes what's already there).
        """
        self.run(f"mkdir -p {remote_dir}")
        key = os.path.expanduser(self.key_file)
        ssh_opt = f"ssh -i {key} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
        for rel in paths:
            local_path = Path(local_dir) / rel
            cmd = ["rsync", "-avz", "--delete", "-e", ssh_opt]
            if local_path.is_dir():
                cmd += [str(local_path) + "/", f"{self.user}@{self.host}:{remote_dir}/{rel}/"]
            else:
                cmd += [str(local_path), f"{self.user}@{self.host}:{remote_dir}/{rel}"]
            subprocess.run(cmd, check=True, capture_output=True)

    def rsync_from(self, remote_path: str, local_dir: str, excludes: list[str] | None = None):
        key = os.path.expanduser(self.key_file)
        cmd = [
            "rsync", "-avz",
            "-e", f"ssh -i {key} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null",
        ]
        for exc in (excludes or []):
            cmd += ["--exclude", exc]
        cmd += [f"{self.user}@{self.host}:{remote_path}", str(local_dir) + "/"]
        subprocess.run(cmd, check=True)


__all__ = ["InstanceHandle"]
