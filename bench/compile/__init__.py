"""bench.compile — builder registry package.

Replaces the old single-module `bench/compile.py`. Build logic now lives behind
a `Builder` ABC + `BuilderRegistry`; the candidate-vs-baseline decision is made
by the orchestration layer (`bench/benchmark.py`) and threaded in as an
`is_baseline` flag.

Back-compat surface kept for the runner and any external importer:
  CompileError, CompileResult, Builder, BuilderRegistry.
"""

from __future__ import annotations

from .builder import (
    BuildError,
    Builder,
    CompileError,
    CompileResult,
)
from .registry import BuilderRegistry


__all__ = [
    "Builder",
    "BuildError",
    "BuilderRegistry",
    "CompileError",
    "CompileResult",
]
