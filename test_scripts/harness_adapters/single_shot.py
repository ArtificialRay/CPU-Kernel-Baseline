"""SingleShotAdapter — one tool-less completion per sample, run in-process."""

from __future__ import annotations

import json
import time
from typing import Optional

from bench.config import BenchmarkConfig
from bench.data.trace_set import TraceSet
from contracts import BASELINE_AUTHORS
from eval.single_shot import run_single_shot
from eval.mcp_client import attach
from skills.launch.launch_session import RemoteTarget

from .base import REPO_ROOT, HarnessAdapter, Job

class SingleShotAdapter(HarnessAdapter):
    """One completion per sample, no tools (eval/single_shot.py::run_single_shot).

    Same shape as OwnHarnessAdapter — in-process, sharing one MCPKernelClient
    across every definition in the batch — but the model never sees a tool
    schema. The MCP session is used only by this adapter, to compile and
    measure what came back, so the number is produced by the same evaluator
    against the same baseline as a multi-turn run and the two are comparable.

    Not compatible with --until-complete: its round planner calls
    _trajectory_complete(), which requires a "submit" turn plus
    --min-iterations exploration calls. Single-shot emits neither (it makes
    2 tool calls per sample and never submits), so every definition would
    look permanently incomplete and the planner would loop forever. Use a
    single pass.
    """

    name = "single-shot"
    # run_single_shot() builds its prompts from the Definition object and never
    # reads job.prompt — placeholder, same as OwnHarnessAdapter.
    prompt_template = "%s"
    template_args = 1

    def __init__(
        self, *, endpoint: str, author: str, remote_root: str, target: RemoteTarget,
        dataset: str, isa: str, model: Optional[str], samples: int = 3,
        temperature: float = 1.0,
    ):
        if not model:
            raise RuntimeError(
                "--model is required for --harness single-shot (a litellm model string, "
                "e.g. anthropic/claude-opus-4-8)."
            )
        self.model = model
        self.dataset = dataset
        self.isa = isa
        self.samples = samples
        self.temperature = temperature
        self.trace_set = TraceSet.from_path(REPO_ROOT / "bench-trace")
        baseline_author = BASELINE_AUTHORS.get(dataset, "reference-scalar")
        self.bench_cfg = BenchmarkConfig(baseline_author=baseline_author)
        self.mcp_client = attach(endpoint, author=author, remote_root=remote_root, target=target)

    def run_job(self, job: Job, *, endpoint: str, author: str, log_path: Path) -> int:
        definition = self.trace_set.definitions[job.name]
        try:
            result = run_single_shot(
                definition=definition,
                trace_set=self.trace_set,
                author=author,
                model=self.model,
                mcp_client=self.mcp_client,
                isa=self.isa,
                dataset=self.dataset,
                bench_cfg=self.bench_cfg,
                samples=self.samples,
                temperature=self.temperature,
                verbose=True,
            )
        except Exception as e:  # noqa: BLE001 — surfaced as a failed job, not a crash
            result = {
                "status": "ERROR",
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "results": [],
            }
        log_path.write_text(json.dumps(result, indent=2))
        # rc drives bench_fleet's retry loop, so it must mean "infra broke",
        # never "the model wrote a kernel that did not compile". A definition
        # where every sample fails IS the measurement here; retrying it would
        # quietly redraw --samples more times and report a pass rate that no
        # longer describes one-shot behaviour. Only the exception path retries.
        return 1 if result.get("status") == "ERROR" else 0

    def cleanup(self) -> None:
        self.mcp_client.close()
