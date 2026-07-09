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

# ── Defaults (single source of truth) ─────────────────────────────────────────

DEFAULT_BASELINE_AUTHOR = "baseline-ncnn-arm"
DEFAULT_WARMUP = 5
DEFAULT_REPEAT = 50
DEFAULT_INNER_ITERS = 1
DEFAULT_CPU = 0
DEFAULT_WATCHDOG_S = 30.0
DEFAULT_CORRECTNESS_ABS_TOL = 1e-3
DEFAULT_CORRECTNESS_REL_TOL = 1e-3
DEFAULT_REQUIRED_MATCHED_RATIO = 1.0
DEFAULT_COLLECT_PERF_COUNTERS = True


@dataclass
class EvalOverride:
    """Sparse per-op-type tolerance overrides for BenchmarkConfig.op_type_config.

    None fields are skipped during merge — only set fields override the
    BenchmarkConfig base values.
    """

    abs_tol: Optional[float] = None
    rel_tol: Optional[float] = None
    required_matched_ratio: Optional[float] = None


# Large fp32 reductions (matmul / attention / MoE) legitimately diverge from a
# numpy reference by more than the elementwise 1e-3 default: ggml accumulates in
# a different SIMD/FMA order, so a few permille of elements drift by ~1e-2..1e-1
# absolute. These op-type overrides make such baselines pass on correctness while
# staying tight enough to catch a real bug. Keyed by Definition.op_type, so they
# only apply to these ops (conv2d / loop_* / rms_norm keep the strict default).
DEFAULT_OP_TYPE_CONFIG: Dict[str, "EvalOverride"] = {
    "gemm": EvalOverride(abs_tol=1e-1, rel_tol=1e-2, required_matched_ratio=0.98),
    "moe":  EvalOverride(abs_tol=1e-1, rel_tol=1e-2, required_matched_ratio=0.98),
    "mha":  EvalOverride(abs_tol=1e-1, rel_tol=1e-2, required_matched_ratio=0.98),
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
