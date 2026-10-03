"""CodexAdapter — local `codex exec` against the same MCP server."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from .base import REPO_ROOT, HarnessAdapter, Job, run_and_tee
from .claude_code import ClaudeCodeAdapter

CODEX_SKILL_FILE = REPO_ROOT / "skills" / "codex" / "codex-kernel-session" / "SKILL.md"
# TOML table key for mcp_servers.<name> in codex's `-c` override — matches the
# mcpServers key ClaudeCodeAdapter uses, just for readability across harness logs.
CODEX_MCP_SERVER_NAME = "cpu-kernel-baseline"
# One single, permanent `--cd` root shared by every codex job, wiped and
# rewritten fresh by CodexAdapter.prepare_workspace() before each run 
CODEX_WORKSPACE_DIR = Path.home() / ".codex-fleet" / "workspace"
# Arbitrary id for the optional custom `model_providers.<id>` entry
# CodexAdapter builds from OPENAI_API_BASE/OPENAI_API_KEY (see its run_job()).
CODEX_DOTENV_PROVIDER_NAME = "dotenv-openai-compatible"

class CodexAdapter(HarnessAdapter):
    """Local Codex CLI (`codex exec`), non-interactive, talking to the same
    MCP server over streamable-http as ClaudeCodeAdapter. Codex has no
    
    `--append-system-prompt` equivalent, but it auto-loads an AGENTS.md from
    its working root (`--cd`) the same way Claude Code auto-loads CLAUDE.md,
    `prepare_workspace()` wiped-and-rewritten-per-run directory rather than 
    a fresh tempdir or a per-job one

    --approve-for-me is just for agent to execute MCP tool without interruption"""

    name = "codex"
    prompt_template = ClaudeCodeAdapter.prompt_template
    template_args = 6

    def __init__(self, *, model: Optional[str]):
        self.model = model
        if not CODEX_SKILL_FILE.exists():
            raise RuntimeError(f"SKILL_FILE not found: {CODEX_SKILL_FILE}")
        if subprocess.run(["which", "codex"], capture_output=True).returncode != 0:
            raise RuntimeError("codex CLI not found on PATH — install Codex CLI first.")
        self.skill_text = CODEX_SKILL_FILE.read_text()
        # Optional: point codex at a custom OpenAI-compatible endpoint instead of
        # whatever account `codex login` already persisted to ~/.codex/auth.json.
        # Per field, first hit wins: OPENAI_API_BASE / OPENAI_BASE_URL /
        # OPENAI_API_KEY in the environment  > `codex login`.
        codex_config = CODEX_SKILL_FILE.parent / "config.json"
        provider = (
            json.loads(codex_config.read_text()).get("model_provider", {})
            if codex_config.exists() else {}
        )
        self.base_url = (
            os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")
            or provider.get("base_url")
        )
        self.api_key = os.environ.get("OPENAI_API_KEY") or provider.get("api_key")

    def run_job(self, job: Job, *, endpoint: str, author: str, log_path: Path) -> int:
        with self.prepare_workspace(job) as workspace:
            cmd = [
                "codex", "exec",
                "--cd", str(workspace),
                "--skip-git-repo-check",
                "--approve-for-me",
                "-c", f'mcp_servers.{CODEX_MCP_SERVER_NAME}.url="{endpoint}"',
                # Codex's default MCP tool_timeout_sec (300s) is shorter than
                # evaluate_kernel()'s own budget, override to 1200s timeout
                "-c", f"mcp_servers.{CODEX_MCP_SERVER_NAME}.tool_timeout_sec=1200",
                "--json",
            ]
            child_env = None
            if self.base_url and self.api_key:
                p = CODEX_DOTENV_PROVIDER_NAME
                # The key reaches only the codex child's environment (which is
                # where env_key below points), not this process or the shell.
                child_env = {**os.environ, "OPENAI_API_KEY": self.api_key}
                cmd += [
                    "-c", f'model_providers.{p}.name="{p}"',
                    "-c", f'model_providers.{p}.base_url="{self.base_url}"',
                    # Value is the *name* of the env var codex reads the key
                    # from at request time — never the key itself.
                    "-c", f'model_providers.{p}.env_key="OPENAI_API_KEY"',
                    # This codex CLI version dropped "chat" wire_api support
                    # entirely (hard config-load error, not a runtime
                    # fallback) — "responses" is the only value it accepts
                    # now: https://github.com/openai/codex/discussions/7782
                    "-c", f'model_providers.{p}.wire_api="responses"',
                    "-c", f'model_provider="{p}"',
                ]
            if self.model:
                cmd += ["-m", self.model]
            cmd.append(job.prompt)
            return run_and_tee(cmd, log_path=log_path, env=child_env)

    @contextmanager
    def prepare_workspace(self, job: Job):
        """Wipe and rewrite the one shared CODEX_WORKSPACE_DIR before every
        run_job() call (every attempt, every job) — nothing accumulates
        between runs. No cleanup_workspace() override needed: the directory is
        reused indefinitely, and the next prepare_workspace() wipes it
        again before its own run anyway."""
        shutil.rmtree(CODEX_WORKSPACE_DIR, ignore_errors=True)
        CODEX_WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
        (CODEX_WORKSPACE_DIR / "AGENTS.md").write_text(self.skill_text)
        yield CODEX_WORKSPACE_DIR
