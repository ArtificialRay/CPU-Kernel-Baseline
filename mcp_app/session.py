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
import re

from contracts import (
    REFERENCE_SCALAR_AUTHORS, REFERENCE_SCALAR_FILENAME, SETUP_HOOK_FILENAME,
    baseline_author_for, setup_weights_for,
)

from .agent_tools import isa as isa_mod
from .agent_tools import resolve_tools
from .agent_tools.base import KernelSession


@dataclass
class SessionConfig:
    dataset: str
    author: str
    isa: str  # required, one of neon/sve/sve2/sme2 (see agent_tools/isa.py)
    bench_trace_root: Path
    run_dir: Path  # session root, e.g. <remote_root>/agent-runs-mcp/<author> — each
    # definition the agent compile()s gets its own run_dir/<definition_name>/ subdir
    baseline_author: Optional[str] = None
    # None = auto-derive from dataset + isa (see contracts.baseline_author_for);
    # only pass explicitly to override.
    instance_label: Optional[str] = None
    max_iterations: Optional[int] = None  # hard per-definition tool-call ceiling; None = unlimited


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

    baseline_author = cfg.baseline_author or baseline_author_for(cfg.dataset, cfg.isa)
    bench_cfg = BenchmarkConfig(baseline_author=baseline_author)

    tools = tools_cls(
        ts, cfg.author, bench_cfg, cfg.run_dir, cfg.isa,
        instance_label=cfg.instance_label,
        max_iterations=cfg.max_iterations,
    )
    _write_reference_scalar_kernels(ts, cfg.dataset, cfg.run_dir)
    return tools


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
        definition = ts.definitions.get(def_name)
        if definition is not None:
            (definition_dir / SETUP_HOOK_FILENAME).write_text(
                _setup_hook_note(definition, ref), encoding="utf-8"
            )


_ENTRY_RE = re.compile(r"int\s+(armbench_entry_\w+)\s*\(([^)]*)\)", re.S)


def _setup_hook_note(definition, ref) -> str:
    """The agent-facing description of the optional untimed setup hook for one
    definition: which inputs it sees for real, and the signature to copy."""
    op = definition.op_type
    weights = setup_weights_for(definition)
    if not weights:
        return (
            f"# Untimed setup: not available for {definition.name}\n\n"
            "This definition has no weight inputs, so `armbench_setup_"
            f"{op}` is never called and every call is timed whole.\n"
        )
    params = None
    for s in ref.sources:
        m = _ENTRY_RE.search(s.content)
        if m and m.group(1) == f"armbench_entry_{op}":
            params = " ".join(m.group(2).split())
            break
    signature = (f'extern "C" int armbench_setup_{op}({params});' if params is not None
                 else f'extern "C" int armbench_setup_{op}(/* same parameters as armbench_entry_{op} */);')
    listed = ", ".join(f"`{w}`" for w in weights)
    return f"""# Optional untimed setup for {definition.name}

Libraries usually prepare weights once (repacking, layout transforms) and
reuse them for every call. Your kernel may do the same, outside the timing,
by exporting two more functions from kernel.cpp:

```cpp
{signature}
extern "C" int armbench_teardown_{op}(void);
```

`armbench_setup_{op}` takes exactly the parameters of `armbench_entry_{op}`,
the function the harness calls on every timed iteration.

- Setup runs once per workload, before the first correctness call and the
  timed loop; teardown runs after the last call. Neither is timed.
- Weights of this definition: {listed}. Setup receives these for real. Every
  other input and the output arrive as zero-filled stand-ins with the same
  shapes; scalar and shape arguments are real. Prepare weights here, never
  results.
- Every timed call to `armbench_entry_{op}` receives the real arguments.
  Keep what setup prepared in static storage and free it in teardown.
  Return 0 on success.
- Both functions are optional. The expert baseline uses the same hook.
"""


__all__ = ["SessionConfig", "build_tools"]
