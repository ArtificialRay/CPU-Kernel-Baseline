"""Job-local Linux filesystem sandbox for external harness CLIs."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class FileSandbox:
    workspace: Path
    runtime_paths: tuple[Path, ...]
    writable_paths: tuple[Path, ...] = ()
    readable_paths: tuple[Path, ...] = ()
    # Relative destination under the private home, e.g. ".codex/auth.json".
    home_files: tuple[tuple[Path, str], ...] = ()


def _outside_repo(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if (resolved == REPO_ROOT or REPO_ROOT in resolved.parents
            or resolved in REPO_ROOT.parents):
        raise RuntimeError(f"Refusing to expose repository path to agent sandbox: {resolved}")
    if not resolved.exists():
        raise RuntimeError(f"Agent sandbox path does not exist: {resolved}")
    return resolved


def _system_mounts() -> list[str]:
    args = ["--ro-bind", "/usr", "/usr"]
    # On merged-/usr systems these are symlinks. Keep their exact targets;
    # other layouts need their real library directories mounted read-only.
    for name in ("bin", "lib", "lib64", "sbin"):
        source = Path("/") / name
        if source.is_symlink():
            args += ["--symlink", os.readlink(source), str(source)]
        elif source.is_dir():
            args += ["--ro-bind", str(source), str(source)]
    for path in (
        "/etc/hosts", "/etc/resolv.conf", "/etc/nsswitch.conf",
        "/etc/passwd", "/etc/group", "/etc/ssl/certs",
    ):
        if Path(path).exists():
            args += ["--ro-bind", path, path]
    return args


def check_file_sandbox(runtime_paths: tuple[Path, ...]) -> None:
    if sys.platform != "linux":
        raise RuntimeError("--file-sandbox bwrap requires Linux")
    if shutil.which("bwrap") is None:
        raise RuntimeError("--file-sandbox bwrap requires bubblewrap (bwrap) on PATH")
    for path in runtime_paths:
        _outside_repo(path)
    probe = subprocess.run(
        ["bwrap", *_system_mounts(), "--proc", "/proc", "--dev", "/dev",
         "--", "/usr/bin/true"],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        raise RuntimeError(f"Bubblewrap cannot start on this host: {probe.stderr.strip()}")


@contextmanager
def sandboxed_command(
    cmd: list[str], spec: FileSandbox, env: dict[str, str] | None = None,
) -> Iterator[tuple[list[str], dict[str, str]]]:
    """Expose only runtime files and this job's paths; never mount the repo."""
    check_file_sandbox(spec.runtime_paths)
    workspace = _outside_repo(spec.workspace)
    executable = shutil.which(cmd[0])
    if executable is None:
        raise RuntimeError(f"Agent CLI not found on PATH: {cmd[0]}")
    executable_path = _outside_repo(Path(executable))
    mounts = [
        "bwrap", "--unshare-pid", "--new-session", "--die-with-parent",
        *_system_mounts(), "--proc", "/proc", "--dev", "/dev",
        "--tmpfs", "/tmp", "--dir", "/home",
    ]
    for path in spec.runtime_paths:
        source = _outside_repo(path)
        mounts += ["--ro-bind", str(source), str(source)]
    for path in spec.readable_paths:
        source = _outside_repo(path)
        mounts += ["--ro-bind", str(source), str(source)]
    for path in (workspace, *spec.writable_paths):
        source = _outside_repo(path)
        mounts += ["--bind", str(source), str(source)]

    with tempfile.TemporaryDirectory(prefix="fleet-agent-home-") as home_dir:
        home = Path(home_dir)
        for source, relative in spec.home_files:
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ValueError(f"Invalid sandbox home path: {relative}")
            source = _outside_repo(source)
            destination = home / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            destination.chmod(0o600)
        mounts += [
            "--bind", str(home), "/home/agent",
            "--setenv", "HOME", "/home/agent",
            "--setenv", "XDG_CONFIG_HOME", "/home/agent/.config",
            "--setenv", "XDG_DATA_HOME", "/home/agent/.local/share",
            "--setenv", "XDG_CACHE_HOME", "/home/agent/.cache",
            "--setenv", "TMPDIR", "/tmp",
            "--chdir", str(workspace), "--",
            str(executable_path), *cmd[1:],
        ]
        child_env = dict(os.environ if env is None else env)
        for key in (
            "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CODEX_HOME",
            "CLAUDE_CONFIG_DIR", "XDG_CONFIG_HOME", "XDG_DATA_HOME",
            "XDG_CACHE_HOME", "TMPDIR", "PWD", "OLDPWD",
        ):
            child_env.pop(key, None)
        runtime_bins: list[str] = []
        for path in spec.runtime_paths:
            resolved = _outside_repo(path)
            if resolved.is_file():
                runtime_bins.append(str(resolved.parent))
            elif (resolved / "bin").is_dir():
                runtime_bins.append(str(resolved / "bin"))
        child_env["PATH"] = os.pathsep.join([*runtime_bins, "/usr/local/bin", "/usr/bin", "/bin"])
        yield mounts, child_env
