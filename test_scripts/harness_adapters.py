"""harness_adapters.py — per-harness HarnessAdapter implementations used by
bench_fleet.py's shared driver. Split out from bench_fleet.py itself so the
orchestration (compute author/label once, provision, prepare_session,
retry/log/sync loop) stays separate from what's genuinely harness-specific:
how each harness is invoked, how it connects to the MCP endpoint, its own
retry quirks.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from bench.config import BenchmarkConfig
from bench.data.trace_set import TraceSet
from contracts import BASELINE_AUTHORS
from eval.evaluator import run_agentic_eval
from eval.mcp_client import attach
from skills.launch.launch_session import RemoteTarget

from dotenv import load_dotenv

load_dotenv()
REPO_ROOT = Path(__file__).resolve().parent.parent

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
CLAUDE_SKILL_FILE = REPO_ROOT / "skills" / "claude-code" / "claude-code-kernel-session" / "SKILL.md"
CODEX_SKILL_FILE = REPO_ROOT / "skills" / "codex" / "codex-kernel-session" / "SKILL.md"
# TOML table key for mcp_servers.<name> in codex's `-c` override — matches the
# mcpServers key ClaudeCodeAdapter uses, just for readability across harness logs.
CODEX_MCP_SERVER_NAME = "cpu-kernel-baseline"
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
    model:str

    @classmethod
    def default_model(cls) -> Optional[str]:
        """The model this harness resolves to when no --model override is
        given
        """
        return None

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
                # timeout: extend tool timeout for claude code to 15 min
                {"mcpServers": {"cpu-kernel-baseline": {
                    "type": "http", "url": endpoint, "timeout": 900000}}},
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


class CodexAdapter(HarnessAdapter):
    """Local Codex CLI (`codex exec`), non-interactive, talking to the same
    MCP server over streamable-http as ClaudeCodeAdapter. Codex has no
    `--append-system-prompt` equivalent, but it auto-loads an AGENTS.md from
    its working root (`--cd`) the same way Claude Code auto-loads CLAUDE.md —
    so each job gets a fresh throwaway dir containing one, instead of a
    system-prompt flag.

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

    def run_job(self, job: Job, *, endpoint: str, author: str, log_path: Path) -> int:
        with tempfile.TemporaryDirectory(prefix="codex-fleet-job-") as job_dir:
            (Path(job_dir) / "AGENTS.md").write_text(self.skill_text)
            cmd = [
                "codex", "exec",
                "--cd", job_dir,
                "--skip-git-repo-check",
                "--approve-for-me",
                "-c", f'mcp_servers.{CODEX_MCP_SERVER_NAME}.url="{endpoint}"',
                # Codex's default MCP tool_timeout_sec (300s) is shorter than
                # evaluate_kernel()'s own budget (750s under the MCP client's
                # 900s tool timeout, see mcp_app/agent_tools/ops.py) — without
                # this override, a slow evaluate() on a big workload times out
                # client-side and the turn is wasted for nothing.
                "-c", f"mcp_servers.{CODEX_MCP_SERVER_NAME}.tool_timeout_sec=900",
                "--json",
            ]
            if self.model:
                cmd += ["-m", self.model]
            cmd.append(job.prompt)
            return _run_and_tee(cmd, log_path=log_path)


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
        return json.loads(NANOBOT_CONFIG_BASE.read_text())["agents"]["defaults"]["model"]

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


class OwnHarnessAdapter(HarnessAdapter):
    """This repo's own litellm agent loop (eval/evaluator.py::run_agentic_eval),
    formerly driven standalone by eval/run_benchmark.py. Unlike
    ClaudeCodeAdapter/NanobotAdapter, there's no external CLI subprocess to
    spawn — run_agentic_eval() already expects an already-connected MCP
    client and runs entirely in-process, so run_job() just calls it
    directly against one MCPKernelClient shared across every job in the
    batch (mcp_app's KernelSession is designed to serve many definitions
    off one long-lived connection — see eval/mcp_client.py's docstring)."""

    name = "own"
    # run_agentic_eval() builds its own system prompt from the Definition object directly and never reads job.prompt,
    # so this is a harmless placeholder, not a real template.
    prompt_template = "%s"
    template_args = 1

    def __init__(
        self, *, endpoint: str, author: str, remote_root: str, target: RemoteTarget,
        dataset: str, isa: str, model: Optional[str], max_turns: int,
    ):
        if not model:
            raise RuntimeError("--model is required for --harness own (a litellm model string).")
        self.model = model
        self.max_turns = max_turns
        self.dataset = dataset
        self.isa = isa
        self.trace_set = TraceSet.from_path(REPO_ROOT / "bench-trace")
        baseline_author = BASELINE_AUTHORS.get(dataset, "reference-scalar")
        self.bench_cfg = BenchmarkConfig(baseline_author=baseline_author)
        self.mcp_client = attach(endpoint, author=author, remote_root=remote_root, target=target)

    def run_job(self, job: Job, *, endpoint: str, author: str, log_path: Path) -> int:
        definition = self.trace_set.definitions[job.name]
        try:
            result = run_agentic_eval(
                definition=definition,
                trace_set=self.trace_set,
                author=author,
                model=self.model,
                mcp_client=self.mcp_client,
                isa=self.isa,
                dataset=self.dataset,
                bench_cfg=self.bench_cfg,
                max_turns=self.max_turns,
                verbose=True,
            )
        except Exception as e:  # noqa: BLE001 — surfaced as a failed job, not a crash
            result = {
                "status": "ERROR",
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "version_history": [],
            }
        log_path.write_text(json.dumps(result, indent=2))
        return 0 if result.get("status") == "PASSED" else 1

    def cleanup(self) -> None:
        self.mcp_client.close()