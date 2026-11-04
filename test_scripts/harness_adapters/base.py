"""Shared pieces of every harness adapter: Job, the HarnessAdapter base class,
and run_and_tee()."""

from __future__ import annotations

import errno
import subprocess
import sys
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from quota_errors import QUOTA_EXIT_CODE, log_has_hard_quota_error

from .sandbox import FileSandbox, sandboxed_command

load_dotenv()
REPO_ROOT = Path(__file__).resolve().parents[2]


class HarnessLaunchError(RuntimeError):
    """The OS temporarily could not start the harness process."""


@dataclass
class Job:
    name: str
    prompt: str
    op_type: str = ""


def run_and_tee(
    cmd: list[str], *, log_path: Path, cwd: Optional[Path] = None, env: Optional[dict] = None,
    sandbox: Optional[FileSandbox] = None,
) -> int:
    """Run `cmd`, streaming its combined stdout/stderr live to the terminal
    while also writing it to log_path (bash's `tee` idiom, ported). `env`, if
    given, is the child's complete environment (None = inherit ours). stdin is
    /dev/null: every harness here takes its prompt as an argument, and a
    non-interactive child must not wait on, or consume, the driver's stdin."""
    context = sandboxed_command(cmd, sandbox, env) if sandbox else nullcontext((cmd, env))
    with context as (child_cmd, child_env), log_path.open("w") as log_fh:
        try:
            proc = subprocess.Popen(
                child_cmd, cwd=cwd, env=child_env, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            )
        except OSError as exc:
            if exc.errno == errno.EAGAIN:
                raise HarnessLaunchError(str(exc)) from exc
            raise
        with proc:
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                log_fh.write(line)
            proc.wait()
            return proc.returncode


class HarnessAdapter:
    """Base adapter; subclasses override what's harness-specific."""

    name: str
    prompt_template: str
    template_args: int  # leading build_jobs() prompt args the template consumes
    model:str
    file_sandbox: str = "off"

    @classmethod
    def default_model(cls) -> Optional[str]:
        """The model this harness resolves to when no --model is given."""
        return None

    def run_job(self, job: Job, *, endpoint: str, author: str, log_path: Path) -> int:
        raise NotImplementedError

    def sandbox_runtime_paths(self) -> tuple[Path, ...]:
        raise NotImplementedError

    def quota_exceeded(self, log_path: Path, return_code: int) -> bool:
        """Adapters may override this for their own structured error format."""
        return log_has_hard_quota_error(log_path)

    def prepare_workspace(self, job: Job) -> AbstractContextManager[Optional[Path]]:
        return nullcontext(None)

    def cleanup_workspace(self, job: Job) -> None:
        """Remove whatever prepare_workspace() created for `job`, if
        anything — called once per job after the whole batch finishes,
        regardless of that job's (or any other job's) success/failure."""
        pass
