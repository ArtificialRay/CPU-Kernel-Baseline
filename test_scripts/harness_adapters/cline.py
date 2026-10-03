"""ClineAdapter — local `cline` CLI against the same MCP server."""

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

CLINE_SKILL_FILE = REPO_ROOT / "skills" / "cline" / "cline-kernel-session" / "SKILL.md"
CLINE_MCP_SERVER_NAME = "cpu-kernel-baseline"
# Same one-shared-directory reasoning as CODEX_WORKSPACE_DIR (see its
# comment) — cwd for cline's own sandboxed shell/file tools, not for the
# system prompt (that goes via `-s`, not an auto-loaded file), wiped and
# recreated fresh by ClineAdapter.prepare_workspace() before each run.
CLINE_WORKSPACE_DIR = Path.home() / ".cline-fleet" / "workspace"

class ClineAdapter(HarnessAdapter):
    name = "cline"
    prompt_template = ClaudeCodeAdapter.prompt_template
    template_args = 6

    def __init__(self, *, model: Optional[str]):
        if not model:
            raise RuntimeError("--model is required for --harness cline (cline auth has no default).")
        self.model = model
        if not CLINE_SKILL_FILE.exists():
            raise RuntimeError(f"SKILL_FILE not found: {CLINE_SKILL_FILE}")
        if subprocess.run(["which", "cline"], capture_output=True).returncode != 0:
            raise RuntimeError("cline CLI not found on PATH — install Cline CLI first.")
        self.skill_text = CLINE_SKILL_FILE.read_text()
        # Per field, first hit wins: OPENAI_API_BASE / OPENAI_BASE_URL /
        # OPENAI_API_KEY in the environment 
        # > the "model_provider" table of the git-ignored config.json next to
        # SKILL.md (copy config.json.example).
        cline_config = CLINE_SKILL_FILE.parent / "config.json"
        provider = (
            json.loads(cline_config.read_text()).get("model_provider", {})
            if cline_config.exists() else {}
        )
        self.base_url = (
            os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")
            or provider.get("base_url")
        )
        self.api_key = os.environ.get("OPENAI_API_KEY") or provider.get("api_key")
        if not self.base_url or not self.api_key:
            raise RuntimeError(
                f"--harness cline needs an endpoint: fill in base_url and api_key in "
                f"{cline_config} (copy config.json.example), or set OPENAI_API_BASE and "
                "OPENAI_API_KEY — unlike codex, cline has no already-logged-in account "
                "to fall back to."
            )

    def run_job(self, job: Job, *, endpoint: str, author: str, log_path: Path) -> int:
        auth_cmd = ["cline", "auth", "-p", self._provider_id(), "-k", self.api_key, "-m", self.model]
        if self._provider_id() == "openai-compatible":
            auth_cmd += ["-b", self.base_url]
        subprocess.run(auth_cmd, check=True, capture_output=True, text=True)
        subprocess.run(
            ["cline", "mcp", "add", CLINE_MCP_SERVER_NAME, endpoint,
             "--transport", "streamable-http", "--yes"],
            check=True, capture_output=True, text=True,
        )
        with self.prepare_workspace(job) as workspace:
            cmd = [
                "cline",
                "-c", str(workspace),
                "-s", self.skill_text,
                "-m", self.model,
                "--auto-approve", "true",
                "--json",
                job.prompt,
            ]
            return run_and_tee(cmd, log_path=log_path)

    def _provider_id(self) -> str:
        """`openai-compatible` hits /v1/chat/completions, which rejects
        gpt-5.6-luna's tool-calling + reasoning_effort combo outright
        (confirmed empirically); `openai-native` hits /v1/responses instead
        and works, but only for a real OpenAI account — so only pick it
        when OPENAI_API_BASE actually points at api.openai.com."""
        return "openai-native" if "api.openai.com" in self.base_url else "openai-compatible"

    @contextmanager
    def prepare_workspace(self, job: Job):
        shutil.rmtree(CLINE_WORKSPACE_DIR, ignore_errors=True)
        CLINE_WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
        yield CLINE_WORKSPACE_DIR
