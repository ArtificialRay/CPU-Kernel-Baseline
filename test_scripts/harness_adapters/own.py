"""OwnHarnessAdapter — this repo's own litellm agent loop, run in-process."""

from __future__ import annotations

import json
import time
from typing import Optional

from bench.config import BenchmarkConfig
from bench.data.trace_set import TraceSet
from contracts import BASELINE_AUTHORS
from eval.evaluator import run_agentic_eval
from eval.mcp_client import attach
from skills.launch.launch_session import RemoteTarget

from .base import REPO_ROOT, HarnessAdapter, Job

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
