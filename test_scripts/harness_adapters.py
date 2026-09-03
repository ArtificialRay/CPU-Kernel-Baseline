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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from bench.config import BenchmarkConfig
from bench.data.trace_set import TraceSet
from contracts import BASELINE_AUTHORS
from eval.evaluator import run_agentic_eval
from eval.mcp_client import attach
from skills.launch.launch_session import RemoteTarget

from dotenv import load_dotenv

load_dotenv()
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


@dataclass
class TurnRow:
    """One turn = one MCP tool_use to the next. `llm_s`/`tool_s` split the
    delta into model-thinking vs remote-compile/evaluate time when the
    harness's log format distinguishes them; harnesses that can't tell them
    apart (no separate 'tool result received' event) leave both at 0.0 and
    report only `total_s`."""
    total_s: float
    llm_s: float = 0.0
    tool_s: float = 0.0


@dataclass
class SessionMetrics:
    """Harness-reported session-level telemetry for one job, normalized
    across harnesses so analysis/wandb_log_run.py never has to know which
    harness produced them. Every field defaults to "unknown" (None/0/empty)
    — a harness that can't report a given field just leaves it at that
    default; consumers must treat absence as "not available", not "zero"."""
    cost_usd: Optional[float] = None
    num_turns: Optional[int] = None
    wall_time_s: Optional[float] = None
    api_retries: int = 0
    session_compile_errors: int = 0
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_cache_read: int = 0
    tokens_cache_created: int = 0
    turn_rows: list[TurnRow] = field(default_factory=list)


def _find(d: Any, key: str) -> Any:
    """Recursive first-match lookup — same idiom as wandb_log_run.py's
    helper of the same name, kept local here since it's a generic ~10-line
    utility, not worth cross-importing between the two modules for."""
    if isinstance(d, dict):
        if d.get(key) is not None:
            return d[key]
        for v in d.values():
            r = _find(v, key)
            if r is not None:
                return r
    elif isinstance(d, list):
        for v in d:
            r = _find(v, key)
            if r is not None:
                return r
    return None


def _parse_ts(s: Optional[str]):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


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

    def parse_session_metrics(self, log_path: Path) -> SessionMetrics:
        """Extract whatever session-level telemetry (cost, tokens, turn
        latency, ...) this harness's log format actually exposes. Default:
        none — a harness whose log doesn't carry structured per-job
        telemetry just returns the all-unknown default, and callers (e.g.
        analysis/wandb_log_run.py) degrade gracefully rather than treating
        that as an error."""
        return SessionMetrics()

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

    def parse_session_metrics(self, log_path: Path) -> SessionMetrics:
        """session metrics logging from JSON event per line at command: `claude -p --output-format stream-json --verbose` 
        including:
            cost_usd, num_turns,
            wall_time_s,api_retries, session_compile_errors,
            tokens_input, tokens_output,
            tokens_cache_read, tokens_cache_created,
            turn_rows
       """
        if not log_path.exists():
            return SessionMetrics()
        cost = turns = dur_ms = None
        retries = compile_errors = 0
        tok_in = tok_out = tok_cache_r = tok_cache_c = 0
        events: list[tuple[str, datetime, str]] = []  # (kind: "llm"|"tool", when, mcp_tool_name)
        for line in log_path.read_text(errors="ignore").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("subtype") == "api_retry":
                retries += 1
            if d.get("type") == "result":
                cost = d.get("total_cost_usd", cost)
                turns = d.get("num_turns", turns)
                dur_ms = d.get("duration_ms", dur_ms)
            u = _find(d, "usage")
            if isinstance(u, dict):
                tok_in += u.get("input_tokens", 0) or 0
                tok_out += u.get("output_tokens", 0) or 0
                tok_cache_r += u.get("cache_read_input_tokens", 0) or 0
                tok_cache_c += u.get("cache_creation_input_tokens", 0) or 0
            when = _parse_ts(d.get("timestamp"))
            content = ((d.get("message") or {}).get("content")) or []
            if when is not None and d.get("type") == "assistant":
                mcp = next((c.get("name", "") for c in content
                            if isinstance(c, dict) and c.get("type") == "tool_use"
                            and str(c.get("name", "")).startswith("mcp__")), "")
                events.append(("llm", when, mcp.split("__")[-1] if mcp else ""))
            for c in content:
                if isinstance(c, dict) and c.get("type") == "tool_result":
                    if when is not None:
                        events.append(("tool", when, ""))
                    t = c.get("content")
                    if isinstance(t, str) and "kernel.cpp" in t and "error" in t.lower():
                        compile_errors += 1
        # fold event deltas into per-turn rows (turn boundary = each MCP tool_use)
        turn_rows: list[TurnRow] = []
        acc = {"llm": 0.0, "tool": 0.0}
        prev_when = prev_boundary = None
        for kind, when, mcp_tool in events:
            if prev_when is not None:
                delta = (when - prev_when).total_seconds()
                if 0 <= delta < 7200:
                    acc[kind] += delta
            prev_when = when
            if kind == "llm" and mcp_tool:
                if prev_boundary is not None:
                    turn_rows.append(TurnRow(
                        total_s=round((when - prev_boundary).total_seconds(), 1),
                        llm_s=round(acc["llm"], 1), tool_s=round(acc["tool"], 1),
                    ))
                prev_boundary, acc = when, {"llm": 0.0, "tool": 0.0}
        return SessionMetrics(
            cost_usd=cost, num_turns=turns,
            wall_time_s=round(dur_ms / 1000.0, 1) if dur_ms else None,
            api_retries=retries, session_compile_errors=compile_errors,
            tokens_input=tok_in, tokens_output=tok_out,
            tokens_cache_read=tok_cache_r, tokens_cache_created=tok_cache_c,
            turn_rows=turn_rows,
        )


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