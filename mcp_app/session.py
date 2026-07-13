"""SessionConfig + build_tools() — server-side session bootstrap.

Runs on the machine mcp_app/server.py is started on (the target instance).
Single chokepoint used by both server.py and the verification script
(mcp_app/smoke_test_driver.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from bench.config import BenchmarkConfig
from bench.data.trace_set import TraceSet

from .agent_tools import isa as isa_mod
from .agent_tools import resolve_tools
from .agent_tools.base import KernelSession

REFERENCE_SCALAR_FILENAME = "reference-scalar-kernel.cpp"


@dataclass
class SessionConfig:
    definition_name: str
    dataset: str
    author: str
    baseline_author: str  # required, no internal dataset->author mapping (see README)
    isa: str  # required, one of neon/sve/sve2/sme2 (see agent_tools/isa.py)
    bench_trace_root: Path
    run_dir: Path  # required, e.g. <remote_root>/agent-runs-mcp/<definition_name>
    instance_label: Optional[str] = None


def build_tools(cfg: SessionConfig) -> KernelSession:
    """Load the TraceSet, resolve the dataset's KernelSession, and construct it.

    Also writes the reference-scalar baseline's kernel.cpp to
    `run_dir/reference-scalar-kernel.cpp` before returning, so it's visible as
    an MCP Resource from the very first `list_resources()` call — before any
    `compile()` has happened. This lets the agent establish a measured
    "naive starting point" baseline as its own first tool call (see
    skills/nanobot/nanobot-kernel-session/SKILL.md).
    """
    ts = TraceSet.from_path(cfg.bench_trace_root)

    definition = ts.definitions.get(cfg.definition_name)
    if definition is None:
        raise ValueError(f"Unknown definition: {cfg.definition_name!r}")

    tools_cls = resolve_tools(cfg.dataset)

    isa_mod.verify_isa_available(cfg.isa)

    bench_cfg = BenchmarkConfig(baseline_author=cfg.baseline_author)

    tools = tools_cls(
        definition, ts, cfg.author, bench_cfg, cfg.run_dir, cfg.isa,
        instance_label=cfg.instance_label,
    )

    _write_reference_scalar_kernel(ts, cfg)

    return tools


def _write_reference_scalar_kernel(ts: TraceSet, cfg: SessionConfig) -> None:
    """Best-effort: not every dataset/definition necessarily has one yet."""
    ref = ts.get_baseline_solution(cfg.definition_name, "reference-scalar")
    if ref is None:
        return
    kernel_src = next((s for s in ref.sources if s.path == "kernel.cpp"), None)
    if kernel_src is None:
        return
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    (cfg.run_dir / REFERENCE_SCALAR_FILENAME).write_text(kernel_src.content, encoding="utf-8")


__all__ = ["SessionConfig", "build_tools", "REFERENCE_SCALAR_FILENAME"]
