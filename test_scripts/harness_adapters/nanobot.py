"""NanobotAdapter — local `nanobot agent` with a per-run patched config."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from .base import REPO_ROOT, HarnessAdapter, Job, run_and_tee

# Overridable so a run can supply its own base config. `--model` alone is not
# enough to switch models across providers: the adapter overrides only
# agents.defaults.model, leaving agents.defaults.provider (and that
# provider's apiKey) pointing at whatever the checked-in config uses, which
# fails with "No API key configured for provider '<other>'".
NANOBOT_CONFIG_BASE = Path(
    os.environ.get(
        "NANOBOT_CONFIG_BASE",
        REPO_ROOT / "skills" / "nanobot" / "nanobot-kernel-session" / "config.json",
    )
)

NANOBOT_WORKSPACE = Path.home() / ".nanobot" / "workspace"
NANOBOT_JOB_WORKSPACES_DIR = Path.home() / ".nanobot" / "job_workspaces"
# nanobot prefers <workspace>/skills/<name>/ over its own stale builtin copy; SKILL.md only 
NANOBOT_SKILL_FILE = REPO_ROOT / "skills" / "nanobot" / "nanobot-kernel-session" / "SKILL.md"
NANOBOT_SKILL_NAME = "nanobot-kernel-session"
# dataset -> the mcpServers key nanobot's config.json wires to a fixed
# local port (always overwritten per-run below, so the base config's port
# value is just a placeholder).
NANOBOT_SERVER_NAME_BY_DATASET = {
    "ncnn": "NCNNKernelBench",
    "llama.cpp": "LLAMACPPKernelBench",
    "simd-loop": "SIMDLoopKernelBench",
    "kleidiai": "KleidiaiKernelBench",
}

class NanobotAdapter(HarnessAdapter):
    name = "nanobot"
    prompt_template = (
        'Optimize the "%s" kernel definition (dataset: %s, baseline solution source: %s) '
        'in new ISA %s. You must spend at least %s compile+evaluate iterations exploring '
        'genuinely different optimization attempts before you are allowed to submit — do not '
        'submit early just because an attempt already looks good, keep iterating until you '
        'hit the floor. You may keep going past it if you are still finding improvements, but '
        'do not exceed %s tool calls total — once you approach that ceiling, stop iterating '
        'and submit your best version immediately, since every iteration spends real model API '
        'budget and the server will start rejecting further compile/evaluate/disassemble calls '
        'once you hit it. Follow the nanobot-kernel-session skill workflow end to end.'
    )
    template_args = 6

    @classmethod
    def default_model(cls) -> Optional[str]:
        if not NANOBOT_CONFIG_BASE.exists():
            raise RuntimeError(
                f"{NANOBOT_CONFIG_BASE} not found — point NANOBOT_CONFIG_BASE at a nanobot "
                "config; the default one is checked in at "
                "skills/nanobot/nanobot-kernel-session/config.json."
            )
        return json.loads(NANOBOT_CONFIG_BASE.read_text())["agents"]["defaults"]["model"]

    def __init__(self, *, dataset: str, model: Optional[str], local_port: int):
        if not NANOBOT_CONFIG_BASE.exists():
            raise RuntimeError(
                f"{NANOBOT_CONFIG_BASE} not found — point NANOBOT_CONFIG_BASE at a nanobot "
                "config; the default one is checked in at "
                "skills/nanobot/nanobot-kernel-session/config.json."
            )
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
        if not NANOBOT_SKILL_FILE.exists():
            raise RuntimeError(f"SKILL_FILE not found: {NANOBOT_SKILL_FILE}")
        server_name = NANOBOT_SERVER_NAME_BY_DATASET.get(dataset)
        if server_name is None:
            raise RuntimeError(
                f"nanobot's config.json has no mcpServers entry for dataset={dataset!r} "
                f"(only {sorted(NANOBOT_SERVER_NAME_BY_DATASET)} are wired today)."
            )
        # Always generate a patched temp config — even with no --model
        # override, the mcpServers URL's port must match this run's actual
        # local_port (nanobot's own MCP client only ever reads it from
        # config, never takes it as a per-invocation argument).
        cfg = json.loads(NANOBOT_CONFIG_BASE.read_text())
        if model:
            cfg["agents"]["defaults"]["model"] = model
            self.model = model
        else:
            self.model = cfg["agents"]["defaults"]["model"]
        cfg["tools"]["mcpServers"][server_name]["url"] = f"http://127.0.0.1:{local_port}/mcp"
        fh = tempfile.NamedTemporaryFile(
            "w", prefix="nanobot-fleet-config-", suffix=".json", delete=False
        )
        json.dump(cfg, fh)
        fh.close()
        self.config_path = Path(fh.name)
        # Set by run_job(); see _job_ws() for why the workspace path needs it.
        self._author: Optional[str] = None

    def run_job(self, job: Job, *, endpoint: str, author: str, log_path: Path) -> int:
        self._author = author
        session = time.strftime("%Y%m%d-%H%M%S")
        with self.prepare_workspace(job) as workspace:
            cmd = [
                "nanobot", "agent", "--logs", "-m", job.prompt,
                "-w", str(workspace), "-c", str(self.config_path), "--session", session,
            ]
            return run_and_tee(cmd, log_path=log_path)

    def _job_ws(self, job: Job) -> Path:
        """Workspace path for `job`, keyed by both author and definition.

        Keying by definition alone caused race conditions when concurrent queues shared
        a definition, leading to `prepare_workspace` (`rsync --delete`) or `cleanup_workspace`
        wiping active session files. Keying by author ensures queue isolation.

        Falls back to the bare definition name if `_author` is unset, ensuring
        `cleanup_workspace()` can safely run on aborted jobs.
        """
        stem = f"{self._author}_{job.name}" if self._author else job.name
        return NANOBOT_JOB_WORKSPACES_DIR / stem

    @contextmanager
    def prepare_workspace(self, job: Job):
        """Per-job workspace isolation so memory/sessions never bleed
        between jobs. Must live outside any git repo — nanobot's GitStore
        refuses to init nested inside one."""
        job_ws = self._job_ws(job)
        job_ws.mkdir(parents=True, exist_ok=True)
        for shared in ("AGENTS.md", "HEARTBEAT.md", "SOUL.md", "USER.md", "prompts", "skills"):
            src = NANOBOT_WORKSPACE / shared
            if src.exists():
                subprocess.run(
                    ["rsync", "-a", "--delete", "--exclude=.git", str(src), f"{job_ws}/"],
                    check=True,
                )
        skill_dir = job_ws / "skills" / NANOBOT_SKILL_NAME
        if skill_dir.is_symlink():
            skill_dir.unlink()
        skill_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(NANOBOT_SKILL_FILE, skill_dir / "SKILL.md")
        yield job_ws

    def cleanup_workspace(self, job: Job) -> None:
        # Must go through _job_ws() too — building at one path and deleting
        # another would leak a workspace per job.
        shutil.rmtree(self._job_ws(job), ignore_errors=True)

    def cleanup(self) -> None:
        self.config_path.unlink(missing_ok=True)
