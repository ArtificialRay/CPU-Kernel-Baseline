"""Benchmark + evaluation configuration (the shared leaf module).

Both config classes live here so `bench/benchmark.py` (orchestration) and
`bench/evaluators/` (evaluation protocol) can each depend on config without
depending on each other.

- `BenchmarkConfig` — the run-level knobs (what to run + how to evaluate).
  Call `BenchmarkConfig.resolve_eval_config(definition)` to get the evaluator
  knobs for a specific definition, including any per-op-type overrides.
- `EvalConfig` — the evaluator-facing subset, fully resolved (no None fields).
- `EvalOverride` — sparse per-op-type overrides (None = inherit from BenchmarkConfig).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from contracts import BASELINE_AUTHORS, EVAL_DEFAULTS, EVAL_OP_TYPE_OVERRIDES

# ── Defaults (single source of truth: config/kernel_contracts.yaml) ───────────

DEFAULT_BASELINE_AUTHOR = BASELINE_AUTHORS["ncnn"]
DEFAULT_WARMUP = EVAL_DEFAULTS["warmup"]
DEFAULT_REPEAT = EVAL_DEFAULTS["repeat"]
DEFAULT_INNER_ITERS = EVAL_DEFAULTS["inner_iters"]
DEFAULT_CPU = EVAL_DEFAULTS["cpu"]
DEFAULT_WATCHDOG_S = EVAL_DEFAULTS["watchdog_s"]
DEFAULT_CORRECTNESS_ABS_TOL = EVAL_DEFAULTS["correctness_abs_tol"]
DEFAULT_CORRECTNESS_REL_TOL = EVAL_DEFAULTS["correctness_rel_tol"]
DEFAULT_REQUIRED_MATCHED_RATIO = EVAL_DEFAULTS["required_matched_ratio"]
DEFAULT_COLLECT_PERF_COUNTERS = EVAL_DEFAULTS["collect_perf_counters"]

@dataclass
class EvalOverride:
    """Sparse per-op-type tolerance overrides for BenchmarkConfig.op_type_config.

    None fields are skipped during merge — only set fields override the
    BenchmarkConfig base values.
    """

    abs_tol: Optional[float] = None
    rel_tol: Optional[float] = None
    required_matched_ratio: Optional[float] = None


# Per-op-type tolerance overrides — see eval_op_type_overrides in
# config/kernel_contracts.yaml for the rationale (fp32 reduction order drift
# in gemm/moe/mha baselines).
DEFAULT_OP_TYPE_CONFIG: Dict[str, "EvalOverride"] = {
    op: EvalOverride(**cfg) for op, cfg in EVAL_OP_TYPE_OVERRIDES.items()
}


@dataclass
class BenchmarkConfig:
    baseline_author: str = DEFAULT_BASELINE_AUTHOR
    definitions: Optional[List[str]] = None
    """If set, only run these definition names."""
    solutions: Optional[List[str]] = None
    """If set, only run these solution names."""
    warmup: int = DEFAULT_WARMUP
    repeat: int = DEFAULT_REPEAT
    inner_iters: int = DEFAULT_INNER_ITERS
    cpu: Optional[int] = DEFAULT_CPU
    abs_tol: float = DEFAULT_CORRECTNESS_ABS_TOL
    rel_tol: float = DEFAULT_CORRECTNESS_REL_TOL
    required_matched_ratio: float = DEFAULT_REQUIRED_MATCHED_RATIO
    min_sqnr_db: float = 20.0
    """SQNR pass threshold (dB) for definitions tagged `correctness:sqnr` (e.g. q8_0
    MoE, whose real quantized arithmetic can't match a full-precision reference on
    an elementwise tolerance). ggml scores ~44 dB; garbage/overflow ~0 dB."""
    op_type_config: Dict[str, EvalOverride] = field(
        default_factory=lambda: dict(DEFAULT_OP_TYPE_CONFIG)
    )
    """Per-op-type tolerance overrides keyed by definition.op_type. Defaults to
    DEFAULT_OP_TYPE_CONFIG (loosened tolerance for gemm/moe/mha float reductions);
    pass an explicit dict (e.g. {}) to opt out."""
    watchdog_s: float = DEFAULT_WATCHDOG_S
    collect_perf_counters: bool = DEFAULT_COLLECT_PERF_COUNTERS

    def resolve_eval_config(self, definition=None) -> "EvalConfig":
        """Merge: BenchmarkConfig base → op_type_config[definition.op_type].

        Higher priority wins. op_type_config is only consulted when definition
        is provided and op_type_config is non-empty.
        """
        atol = self.abs_tol
        rtol = self.rel_tol
        ratio = self.required_matched_ratio
        if definition is not None and self.op_type_config:
            op = self.op_type_config.get(definition.op_type)
            if op is not None:
                if op.abs_tol is not None:
                    atol = op.abs_tol
                if op.rel_tol is not None:
                    rtol = op.rel_tol
                if op.required_matched_ratio is not None:
                    ratio = op.required_matched_ratio
        return EvalConfig(
            abs_tol=atol,
            rel_tol=rtol,
            required_matched_ratio=ratio,
            min_sqnr_db=self.min_sqnr_db,
            warmup=self.warmup,
            repeat=self.repeat,
            inner_iters=self.inner_iters,
            cpu=self.cpu,
            watchdog_s=self.watchdog_s,
            collect_perf_counters=self.collect_perf_counters,
            baseline_author=self.baseline_author,
        )


@dataclass(frozen=True)
class EvalConfig:
    """Fully resolved evaluator knobs.

    Produced by BenchmarkConfig.resolve_eval_config(definition). All fields are
    concrete — no None values, no op_type lookup needed downstream.
    """

    # correctness
    abs_tol: float = DEFAULT_CORRECTNESS_ABS_TOL
    rel_tol: float = DEFAULT_CORRECTNESS_REL_TOL
    required_matched_ratio: float = DEFAULT_REQUIRED_MATCHED_RATIO
    min_sqnr_db: float = 20.0
    # timing
    warmup: int = DEFAULT_WARMUP
    repeat: int = DEFAULT_REPEAT
    inner_iters: int = DEFAULT_INNER_ITERS
    cpu: Optional[int] = DEFAULT_CPU
    watchdog_s: float = DEFAULT_WATCHDOG_S
    # perf
    collect_perf_counters: bool = DEFAULT_COLLECT_PERF_COUNTERS
    # speedup
    baseline_author: str = DEFAULT_BASELINE_AUTHOR


__all__ = [
    "BenchmarkConfig",
    "EvalConfig",
    "EvalOverride",
    "DEFAULT_BASELINE_AUTHOR",
    "DEFAULT_WARMUP",
    "DEFAULT_REPEAT",
    "DEFAULT_INNER_ITERS",
    "DEFAULT_CPU",
    "DEFAULT_WATCHDOG_S",
    "DEFAULT_CORRECTNESS_ABS_TOL",
    "DEFAULT_CORRECTNESS_REL_TOL",
    "DEFAULT_REQUIRED_MATCHED_RATIO",
    "DEFAULT_COLLECT_PERF_COUNTERS",
]
