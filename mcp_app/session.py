"""SessionConfig + build_tools() — server-side session bootstrap.

Runs on the machine mcp_app/server.py is started on (the target instance).
Single chokepoint used by both server.py and the verification script
(mcp_app/smoke_test_driver.py).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from bench.config import BenchmarkConfig
from bench.data.trace_set import TraceSet
from contracts import REFERENCE_SCALAR_AUTHORS, REFERENCE_SCALAR_FILENAME

from .agent_tools import isa as isa_mod
from .agent_tools import resolve_tools
from .agent_tools.base import KernelSession
from .agent_tools.baseline_readiness import DEFAULT_BASELINE_AUTHOR


@dataclass
class SessionConfig:
    dataset: str
    author: str
    isa: str  # required, one of neon/sve/sve2/sme2 (see agent_tools/isa.py)
    bench_trace_root: Path
    run_dir: Path  # session root, e.g. <remote_root>/agent-runs-mcp/<author> — each
    # definition the agent compile()s gets its own run_dir/<definition_name>/ subdir
    baseline_author: Optional[str] = None
    # None = auto-derive from dataset (see baseline_readiness.DEFAULT_BASELINE_AUTHOR);
    # only pass explicitly to override.
    instance_label: Optional[str] = None


def build_tools(cfg: SessionConfig) -> KernelSession:
    """Load the TraceSet, resolve the dataset's KernelSession, and construct it.

    No single `definition` is resolved here — `KernelSession` is
    multi-definition; each `compile(definition=..., code=...)` call resolves
    (and lazily runs the — potentially slow — baseline check for) its own
    definition the first time it's touched. See agent_tools/base.py.

    reference-scalar-kernel.cpp is different: it must be readable via
    list_resources()/read_resource() *before* the agent's first compile()
    call for a definition (so it can compile that as v1), so it can't be
    lazy the same way — write every definition's up front instead (cheap:
    pure text-file I/O, no compile/evaluate), see
    `_write_reference_scalar_kernels` below.
    """
    ts = TraceSet.from_path(cfg.bench_trace_root)

    tools_cls = resolve_tools(cfg.dataset)

    isa_mod.verify_isa_available(cfg.isa)

    baseline_author = cfg.baseline_author or DEFAULT_BASELINE_AUTHOR[cfg.dataset]
    bench_cfg = BenchmarkConfig(baseline_author=baseline_author)

    tools = tools_cls(
        ts, cfg.author, bench_cfg, cfg.run_dir, cfg.isa,
        instance_label=cfg.instance_label,
    )
    _write_reference_scalar_kernels(ts, cfg.dataset, cfg.run_dir)
    _write_hardware_docs(cfg.run_dir)
    return tools


# Arm Software Optimization Guides bundled with the server, exposed to every
# session as read-on-demand MCP resources under `<run_dir>/docs/` (see
# resources.py::_DOCS_DIRNAME). Static + dataset-independent, so written once at
# run_dir root rather than per-definition.
_HARDWARE_DOCS_SRC = Path(__file__).parent / "hardware_docs"
DOCS_DIRNAME = "docs"


def _write_hardware_docs(run_dir: Path) -> None:
    """Copy the bundled Arm optimization guides into `run_dir/docs/*.md` so the
    agent can read per-instruction latency/throughput tables via list_resources()
    /read_resource() — the same MCP path as the kernel files. Best-effort."""
    if not _HARDWARE_DOCS_SRC.is_dir():
        return
    dest = run_dir / DOCS_DIRNAME
    dest.mkdir(parents=True, exist_ok=True)
    for md in sorted(_HARDWARE_DOCS_SRC.glob("*.md")):
        try:
            shutil.copyfile(md, dest / md.name)
        except OSError:
            pass


def _write_reference_scalar_kernels(ts: TraceSet, dataset: str, run_dir: Path) -> None:
    """Write every this-dataset definition's reference-scalar kernel.cpp to
    `run_dir/<definition_name>/reference-scalar-kernel.cpp`, best-effort
    (not every definition necessarily has one yet).
    """
    def_names = {
        def_name
        for def_name, sols in ts.solutions.items()
        if any(s.dataset.value == dataset for s in sols)
    }
    # Dataset-dependent (simd-loop's solution author is "reference", not
    # "reference-scalar" — see contracts.py). A single flat literal here
    # silently dropped every simd-loop definition's starter kernel.
    ref_author = REFERENCE_SCALAR_AUTHORS[dataset]
    for def_name in def_names:
        ref = ts.get_baseline_solution(def_name, ref_author)
        if ref is None:
            continue
        kernel_src = next((s for s in ref.sources if s.path == "kernel.cpp"), None)
        if kernel_src is None:
            continue
        definition_dir = run_dir / def_name
        definition_dir.mkdir(parents=True, exist_ok=True)
        (definition_dir / REFERENCE_SCALAR_FILENAME).write_text(
            kernel_src.content, encoding="utf-8"
        )


__all__ = ["SessionConfig", "build_tools"]
