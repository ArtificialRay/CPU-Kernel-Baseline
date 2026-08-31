"""harness_adapters.py — per-harness HarnessAdapter implementations used by
bench_fleet.py's shared driver. Split out from bench_fleet.py itself so the
orchestration (compute author/label once, provision, prepare_session,
retry/log/sync loop) stays separate from what's genuinely harness-specific:
how each harness is invoked, how it connects to the MCP endpoint, its own
retry quirks.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

NANOBOT_CONFIG_BASE = REPO_ROOT / "skills" / "nanobot" / "nanobot-kernel-session" / "config.json"
CLAUDE_SKILL_FILE = REPO_ROOT / "skills" / "claude-code" / "claude-code-kernel-session" / "SKILL.md"
NANOBOT_WORKSPACE = Path.home() / ".nanobot" / "workspace"
NANOBOT_JOB_WORKSPACES_DIR = Path.home() / ".nanobot" / "job_workspaces"

# dataset -> the mcpServers key nanobot's config.json wires to a fixed
# local port (always overwritten per-run below, so the base config's port
# value is just a placeholder).
NANOBOT_SERVER_NAME_BY_DATASET = {
    "ncnn": "NCNNKernelBench",
    "llama.cpp": "LLAMACPPKernelBench",
    "simd-loop": "SIMDLoopKernelBench",
}


@dataclass
class Job:
    name: str
    prompt: str


def _run_and_tee(cmd: list[str], *, log_path: Path, cwd: Optional[Path] = None) -> int:
    """Run `cmd`, streaming its combined stdout/stderr live to the terminal
    while also writing it to log_path (bash's `tee` idiom, ported)."""
    with log_path.open("w") as log_fh, subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
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

    def run_job(self, job: Job, *, endpoint: str, author: str, log_path: Path) -> int:
        raise NotImplementedError

    def is_benign_failure(self, log_path: Path) -> bool:
        return False

    def prepare_workspace(self, job: Job) -> AbstractContextManager[Optional[Path]]:
        return nullcontext(None)

    def cleanup_workspace(self, job: Job) -> None:
        """Remove whatever prepare_workspace() created for `job`, if
        anything — called once per job after the whole batch finishes,
        regardless of that job's (or any other job's) success/failure."""
        pass


class ClaudeCodeAdapter(HarnessAdapter):
    name = "claude-code"
    prompt_template = (
        'Optimize the "%s" kernel definition (dataset: %s, baseline solution source: %s) '
        'in ISA %s. You must spend at least %s tool calls but not exceed %s tool calls to '
        'explore genuinely different optimization attempts before you are allowed to submit. '
        'once you hit that ceiling, stop iterating and submit your best version immediately, '
        'since every iteration spends real model API budget. Follow the ground rules and '
        'workflow in your system prompt.'
    )
    template_args = 6

    def __init__(self, *, model: Optional[str], max_budget_usd: Optional[str]):
        self.model = model
        self.max_budget_usd = max_budget_usd
        if not CLAUDE_SKILL_FILE.exists():
            raise RuntimeError(f"SKILL_FILE not found: {CLAUDE_SKILL_FILE}")
        if subprocess.run(["which", "claude"], capture_output=True).returncode != 0:
            raise RuntimeError("claude CLI not found on PATH — install Claude Code first.")
        self.system_prompt = CLAUDE_SKILL_FILE.read_text()

    def run_job(self, job: Job, *, endpoint: str, author: str, log_path: Path) -> int:
        with tempfile.NamedTemporaryFile(
            "w", prefix="claude-fleet-mcp-", suffix=".json", delete=False
        ) as mcp_config_fh:
            json.dump(
                {"mcpServers": {"cpu-kernel-baseline": {"type": "http", "url": endpoint}}},
                mcp_config_fh,
            )
            mcp_config_path = Path(mcp_config_fh.name)
        try:
            cmd = [
                "claude", "-p",
                "--mcp-config", str(mcp_config_path),
                "--strict-mcp-config",
                "--permission-mode", "bypassPermissions",
                "--disallowedTools", "Bash", "Task", "WebFetch", "WebSearch",
                "--append-system-prompt", self.system_prompt,
                "--no-session-persistence",
                "--output-format", "stream-json",
                "--verbose",
            ]
            if self.model:
                cmd += ["--model", self.model]
            if self.max_budget_usd:
                cmd += ["--max-budget-usd", self.max_budget_usd]
            cmd.append(job.prompt)
            return _run_and_tee(cmd, log_path=log_path)
        finally:
            mcp_config_path.unlink(missing_ok=True)


class NanobotAdapter(HarnessAdapter):
    name = "nanobot"
    prompt_template = (
        'Optimize the "%s" kernel definition (dataset: %s, baseline solution source: %s) '
        'in new ISA %s. You must spend at least %s compile+evaluate iterations exploring '
        'genuinely different optimization attempts before you are allowed to submit — do not '
        'submit early just because an attempt already looks good, keep iterating until you '
        'hit the floor. You may keep going past it if you are still finding improvements. '
        'Follow the nanobot-kernel-session skill workflow end to end.'
    )
    template_args = 5

    def __init__(self, *, dataset: str, model: Optional[str], local_port: int):
        if subprocess.run(["which", "nanobot"], capture_output=True).returncode != 0:
            raise RuntimeError(
                "nanobot CLI not found on PATH — pip install nanobot-ai (pinned in "
                "requirements.txt) first."
            )
        if not NANOBOT_WORKSPACE.exists():
            raise RuntimeError(
                f"GLOBAL_WORKSPACE ({NANOBOT_WORKSPACE}) doesn't exist yet — run "
                "'nanobot agent -m \"hi\"' once to bootstrap AGENTS.md/SOUL.md/skills/ "
                "before using this script."
            )
        server_name = NANOBOT_SERVER_NAME_BY_DATASET.get(dataset)
        if server_name is None:
            raise RuntimeError(
                f"nanobot's config.json has no mcpServers entry for dataset={dataset!r} "
                f"(only {sorted(NANOBOT_SERVER_NAME_BY_DATASET)} are wired today)."
            )
        self.model = model
        # Always generate a patched temp config — even with no --model
        # override, the mcpServers URL's port must match this run's actual
        # local_port (nanobot's own MCP client only ever reads it from
        # config, never takes it as a per-invocation argument).
        cfg = json.loads(NANOBOT_CONFIG_BASE.read_text())
        if model:
            cfg["agents"]["defaults"]["model"] = model
        cfg["tools"]["mcpServers"][server_name]["url"] = f"http://127.0.0.1:{local_port}/mcp"
        fh = tempfile.NamedTemporaryFile(
            "w", prefix="nanobot-fleet-config-", suffix=".json", delete=False
        )
        json.dump(cfg, fh)
        fh.close()
        self.config_path = Path(fh.name)

    def run_job(self, job: Job, *, endpoint: str, author: str, log_path: Path) -> int:
        session = time.strftime("%Y%m%d-%H%M%S")
        with self.prepare_workspace(job) as workspace:
            cmd = [
                "nanobot", "agent", "--logs", "-m", job.prompt,
                "-w", str(workspace), "-c", str(self.config_path), "--session", session,
            ]
            return _run_and_tee(cmd, log_path=log_path)

    def is_benign_failure(self, log_path: Path) -> bool:
        """Known benign nanobot bug: close_mcp() can crash on
        CancelledError after the job's own work (and evaluate()'s
        auto-persist) already finished — treat as success, not a retry."""
        text = log_path.read_text(errors="replace")
        return "asyncio.exceptions.CancelledError" in text and "close_mcp" in text

    @contextmanager
    def prepare_workspace(self, job: Job):
        """Per-job workspace isolation so memory/sessions never bleed
        between jobs. Must live outside any git repo — nanobot's GitStore
        refuses to init nested inside one."""
        job_ws = NANOBOT_JOB_WORKSPACES_DIR / job.name
        job_ws.mkdir(parents=True, exist_ok=True)
        for shared in ("AGENTS.md", "HEARTBEAT.md", "SOUL.md", "USER.md", "prompts", "skills"):
            src = NANOBOT_WORKSPACE / shared
            if src.exists():
                subprocess.run(
                    ["rsync", "-a", "--delete", "--exclude=.git", str(src), f"{job_ws}/"],
                    check=True,
                )
        yield job_ws

    def cleanup_workspace(self, job: Job) -> None:
        shutil.rmtree(NANOBOT_JOB_WORKSPACES_DIR / job.name, ignore_errors=True)

    def cleanup(self) -> None:
        self.config_path.unlink(missing_ok=True)
