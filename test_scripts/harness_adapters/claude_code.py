"""ClaudeCodeAdapter — local `claude -p`, sandboxed to a per-job empty cwd."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from .base import REPO_ROOT, HarnessAdapter, Job, run_and_tee

CLAUDE_SKILL_FILE = REPO_ROOT / "skills" / "claude-code" / "claude-code-kernel-session" / "SKILL.md"
# mcpServers key in the --mcp-config ClaudeCodeAdapter writes, and the server
# part of the `mcp__<name>` allow rule below — the two must stay in sync.
CLAUDE_MCP_SERVER_NAME = "cpu-kernel-baseline"
# Optional parent for ClaudeCodeAdapter's per-job working directories
# (default: the system temp dir). Must be outside this repo — see
# ClaudeCodeAdapter.prepare_workspace().
CLAUDE_JOB_ROOT_ENV = "ARMBENCH_CC_JOB_ROOT"
# Everything a claude-code job may do: this session's MCP tools, plus file
# tools confined to its own working directory. `--permission-mode dontAsk`
# denies whatever is not listed here, including any path outside cwd.
CLAUDE_ALLOWED_TOOLS = (
    f"mcp__{CLAUDE_MCP_SERVER_NAME}", "ListMcpResourcesTool", "ReadMcpResourceTool",
    "Read(./**)", "Write(./**)", "Edit(./**)", "Glob(./**)", "Grep(./**)",
)
# Dropped from the model's tool list altogether (dontAsk would deny them
# anyway): shell, network, sub-agents, and anything that outlives the job.
CLAUDE_DISALLOWED_TOOLS = (
    "Bash", "Task", "Agent", "WebFetch", "WebSearch", "NotebookEdit",
    "Skill", "SendMessage", "ListAgents", "Workflow", "Monitor",
    "CronCreate", "CronDelete", "CronList", "ScheduleWakeup",
    "EnterWorktree", "ExitWorktree", "PushNotification",
)

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

    def __init__(self, *, model: Optional[str]):
        self.model = model
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
                # timeout: extend tool timeout for claude code to 15 min
                {"mcpServers": {CLAUDE_MCP_SERVER_NAME: {
                    "type": "http", "url": endpoint, "timeout": 900000}}},
                mcp_config_fh,
            )
            mcp_config_path = Path(mcp_config_fh.name)
        try:
            with self.prepare_workspace(job) as workspace:
                cmd = [
                    "claude", "-p",
                    "--mcp-config", str(mcp_config_path),
                    "--strict-mcp-config",
                    "--permission-mode", "dontAsk",
                    "--allowedTools", *CLAUDE_ALLOWED_TOOLS,
                    "--disallowedTools", *CLAUDE_DISALLOWED_TOOLS,
                    "--settings", json.dumps({"autoMemoryEnabled": False}),
                    "--append-system-prompt", self.system_prompt,
                    "--no-session-persistence",
                    "--output-format", "stream-json",
                    "--verbose",
                ]
                if self.model:
                    cmd += ["--model", self.model]
                cmd.append(job.prompt)
                # Auto-memory is off twice over (the setting above and this
                # variable): notes written by one job must never reach another.
                env = {**os.environ, "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"}
                return run_and_tee(cmd, log_path=log_path, cwd=workspace, env=env)
        finally:
            mcp_config_path.unlink(missing_ok=True)

    @contextmanager
    def prepare_workspace(self, job: Job):
        """Fresh, empty working directory for one run_job() call, outside
        this repo, removed again afterwards. So non of benchmarking scaffold or
        dataset would be seen."""
        root = os.environ.get(CLAUDE_JOB_ROOT_ENV) or None
        if root is not None:
            root_path = Path(root).expanduser().resolve()
            if root_path == REPO_ROOT or REPO_ROOT in root_path.parents:
                # Claude Code also loads CLAUDE.md from every parent of cwd.
                raise RuntimeError(
                    f"{CLAUDE_JOB_ROOT_ENV}={root} is inside this repo; claude-code "
                    "jobs must run outside it."
                )
            root_path.mkdir(parents=True, exist_ok=True)
            root = str(root_path)
        with tempfile.TemporaryDirectory(
            prefix=f"cc-job-{job.name}-", dir=root, ignore_cleanup_errors=True,
        ) as job_dir:
            yield Path(job_dir)
