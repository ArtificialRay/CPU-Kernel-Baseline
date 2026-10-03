"""Shared pieces of every harness adapter: Job, the HarnessAdapter base class,
and run_and_tee()."""

from __future__ import annotations

import subprocess
import sys
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Job:
    name: str
    prompt: str


def run_and_tee(
    cmd: list[str], *, log_path: Path, cwd: Optional[Path] = None, env: Optional[dict] = None,
) -> int:
    """Run `cmd`, streaming its combined stdout/stderr live to the terminal
    while also writing it to log_path (bash's `tee` idiom, ported). `env`, if
    given, is the child's complete environment (None = inherit ours). stdin is
    /dev/null: every harness here takes its prompt as an argument, and a
    non-interactive child must not wait on, or consume, the driver's stdin."""
    with log_path.open("w") as log_fh, subprocess.Popen(
        cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_fh.write(line)
        proc.wait()
        return proc.returncode


class HarnessAdapter:
    """Base adapter — ClaudeCodeAdapter/NanobotAdapter override what's
    genuinely harness-specific. own-harness (Step 3) adds a third."""

    name: str
    prompt_template: str
    template_args: int  # 5 or 6 — see bench_fleet.py::build_jobs' docstring
    model:str

    @classmethod
    def default_model(cls) -> Optional[str]:
        """The model this harness resolves to when no --model override is
        given
        """
        return None

    def run_job(self, job: Job, *, endpoint: str, author: str, log_path: Path) -> int:
        raise NotImplementedError

    def prepare_workspace(self, job: Job) -> AbstractContextManager[Optional[Path]]:
        return nullcontext(None)

    def cleanup_workspace(self, job: Job) -> None:
        """Remove whatever prepare_workspace() created for `job`, if
        anything — called once per job after the whole batch finishes,
        regardless of that job's (or any other job's) success/failure."""
        pass
