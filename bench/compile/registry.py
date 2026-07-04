"""BuilderRegistry — singleton dispatch + build cache for arm-bench.

Mirrors flashinfer-bench's registry shape (singleton, priority list, hash cache,
filelock-serialized builds), adapted for arm-bench:

  - the cache key is `(solution.hash(), is_baseline)` — the same solution built
    as a baseline (NcnnBuilder) vs as a candidate (CandidateBuilder) are distinct
    artifacts;
  - the cached value is a `CompileResult` (a `.so` on disk), not a Runnable;
  - `cleanup()` rmtree's every cached build_dir. Lifecycle is owned by
    `Benchmark` (builds during run_all, frees in `Benchmark.close()`), so the
    runner no longer cleans up per call.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Tuple, Type

import filelock

from bench.data.definition import Definition
from bench.data.solution import Solution

from .builder import BuildError, Builder, CompileResult
from .builders import CandidateBuilder, LlamaCppBuilder, NcnnBuilder, SimdLoopBuilder

_BUILDER_PRIORITY: List[Type[Builder]] = [
    CandidateBuilder,
    NcnnBuilder,
    SimdLoopBuilder,
    LlamaCppBuilder,
]
"""Builder types in priority order for automatic selection."""

_LOCK_DIR = Path(tempfile.gettempdir()) / "armbench_build_locks"


class BuilderRegistry:
    """Central registry for dispatching and caching builds (singleton)."""

    _instance: ClassVar[Optional["BuilderRegistry"]] = None
    _builders: List[Builder]
    _cache: Dict[Tuple[str, bool], "CompileResult | BuildError"]

    def __init__(self, builders: List[Builder]) -> None:
        if len(builders) == 0:
            raise ValueError("BuilderRegistry requires at least one builder")
        self._builders = list(builders)
        self._cache = {}

    @classmethod
    def get_instance(cls) -> "BuilderRegistry":
        """Get the singleton, instantiating every available builder in priority order."""
        if cls._instance is None:
            builders: List[Builder] = []
            for builder_type in _BUILDER_PRIORITY:
                if builder_type.is_available():
                    builders.append(builder_type())
            cls._instance = BuilderRegistry(builders)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Drop the singleton (mainly for tests)."""
        cls._instance = None

    def build(
        self, definition: Definition, solution: Solution, is_baseline: bool
    ) -> CompileResult:
        """Build `solution` into a `.so`, using the cache if available.

        Selects the first builder whose `can_build(solution, is_baseline)` is True.
        Process-safe: concurrent builds of the same key are serialized via filelock.
        """
        key = (solution.hash(), is_baseline)
        cached = self._cache.get(key)
        if cached is not None:
            if isinstance(cached, BuildError):
                raise cached
            if cached.so_path.exists():
                return cached
            # Artifact was cleaned up out from under us — rebuild.
            del self._cache[key]

        _LOCK_DIR.mkdir(parents=True, exist_ok=True)
        lock_file = _LOCK_DIR / f"{solution.hash()}-{int(is_baseline)}.lock"
        with filelock.FileLock(str(lock_file)):
            cached = self._cache.get(key)
            if cached is not None:
                if isinstance(cached, BuildError):
                    raise cached
                if cached.so_path.exists():
                    return cached
                del self._cache[key]

            builder = next(
                (b for b in self._builders if b.can_build(solution, is_baseline)), None
            )
            if builder is None:
                err = BuildError(
                    f"No registered builder can build solution '{solution.name}' "
                    f"(dataset={solution.dataset.value}, is_baseline={is_baseline})"
                )
                self._cache[key] = err
                raise err

            result = builder.build(definition, solution)
            self._cache[key] = result
            return result

    def cleanup(self) -> None:
        """Remove every cached build_dir and clear the cache."""
        for cached in self._cache.values():
            if isinstance(cached, CompileResult):
                shutil.rmtree(cached.build_dir, ignore_errors=True)
        self._cache.clear()


__all__ = ["BuilderRegistry"]
