"""bench.compile — builder registry package.

Replaces the old single-module `bench/compile.py`. Build logic now lives behind
a `Builder` ABC + `BuilderRegistry`; the candidate-vs-baseline decision is made
by the orchestration layer (`bench/benchmark.py`) and threaded in as an
`is_baseline` flag.

Back-compat surface kept for the runner and any external importer:
  CompileError, CompileResult, cleanup_build_dir, Builder, BuilderRegistry.
"""

from __future__ import annotations

import shutil

from .builder import (
    ARM_BENCH_ROOT,
    BENCH_ROOT,
    BuildError,
    Builder,
    CompileError,
    CompileResult,
)
from .registry import BuilderRegistry


def cleanup_build_dir(result: CompileResult) -> None:
    """Remove a single CompileResult's build dir. Idempotent.

    Lifecycle is normally owned by BuilderRegistry.cleanup() / Benchmark.close();
    this helper remains for callers that build a one-off result outside the
    registry.
    """
    if result.build_dir.exists():
        shutil.rmtree(result.build_dir, ignore_errors=True)


__all__ = [
    "ARM_BENCH_ROOT",
    "BENCH_ROOT",
    "Builder",
    "BuildError",
    "BuilderRegistry",
    "CompileError",
    "CompileResult",
    "cleanup_build_dir",
]
