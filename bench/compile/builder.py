"""Builder ABC + CompileError/CompileResult + shared build helpers.

A Builder turns a Solution into a `.so` for dlopen. The concrete subclasses
live in `bench/compile/builders/`:

  - CandidateBuilder : shared raw-`float*` path for ALL candidate (non-baseline)
                       solutions, regardless of dataset. No ncnn dependency.
  - NcnnBuilder      : per-dataset baseline path (dataset=ncnn). Ports the old
                       compile_solution: ncnn framework sources + _mat_factory +
                       stubs, retargeted to the real `ncnn/src` + `ncnn/src/layer/arm`
                       checkout layout.
  - SimdLoopBuilder  : per-dataset baseline stub (dataset=simd-loop).

The candidate-vs-baseline decision is made by the orchestration layer
(`bench/benchmark.py`) by comparing `solution.author == baseline_author` and
passing the resulting `is_baseline` flag into `BuilderRegistry.build(...)`,
which selects the builder via `can_build(solution, is_baseline)`.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from bench.data.definition import Definition
from bench.data.solution import Solution

logger = logging.getLogger(__name__)


# ── Paths ────────────────────────────────────────────────────────────────────
#
# This file is bench/compile/builder.py, so:
#   parent        → bench/compile
#   parent.parent → bench
#   parent³       → arm-bench root (== the flattened cpu-kernel-baseline/ repo)
#
# NOTE (deviation from plan): the original plan assumed arm-bench lived at
# cpu-kernel-baseline/arm-bench/ and therefore advised arm_bench_root.parent.parent
# for the ncnn checkout. In the current (flattened) layout arm-bench IS
# cpu-kernel-baseline/, the real ncnn checkout sits at the repo root next to it,
# so `ARM_BENCH_ROOT.parent / "ncnn"` is already correct. Verified against the
# on-disk checkout (ncnn/src/*.cpp, ncnn/src/layer/arm/).

BENCH_ROOT = Path(__file__).resolve().parent.parent
ARM_BENCH_ROOT = BENCH_ROOT.parent


def solutions_root(arm_bench_root: Path) -> Path:
    """Warehouse solutions dir; the `_harness/<op>.{cpp,h}` files hang off this."""
    return arm_bench_root / "bench-trace" / "solutions"


# ── Build error ──────────────────────────────────────────────────────────────

class CompileError(RuntimeError):
    """clang++ returned non-zero. The message includes the full stderr."""

    def __init__(self, message: str, returncode: int, stderr: str, command: List[str]):
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr
        self.command = command


# Alias matching flashinfer-bench's naming, used by the registry for
# "no builder can build this" / "build failed" conditions.
class BuildError(RuntimeError):
    """No registered builder can build a solution, or a build failed structurally."""


# ── Compile result ───────────────────────────────────────────────────────────

@dataclass
class CompileResult:
    so_path: Path
    """Absolute path to the produced .so."""
    build_dir: Path
    """Temp dir holding the build artifacts. Lifecycle owned by the registry /
    Benchmark.close(); the runner no longer cleans up per call."""
    command: List[str]
    """The clang++ command that was run."""


# ── Builder ABC ──────────────────────────────────────────────────────────────

class Builder(ABC):
    """Base class for all builders.

    Subclasses declare a `_build_dir_name` prefix (for tempdir naming) and
    implement is_available / can_build / build. The shared helpers below lift
    the boilerplate that used to live inline in `compile_solution`.
    """

    _build_dir_name: str

    def __init__(self, build_dir_name: str) -> None:
        self._build_dir_name = build_dir_name

    @staticmethod
    @abstractmethod
    def is_available() -> bool:
        """True if this builder's toolchain is present on the host."""

    @abstractmethod
    def can_build(self, solution: Solution, is_baseline: bool) -> bool:
        """True if this builder should build `solution` given the is_baseline flag."""

    @abstractmethod
    def build(self, definition: Definition, solution: Solution) -> CompileResult:
        """Compile `solution` into a `.so` and return its CompileResult."""

    # ── shared helpers ────────────────────────────────────────────────────────

    def _make_build_dir(
        self, solution: Solution, build_dir: Optional[Path] = None
    ) -> Tuple[Path, Path]:
        """Create (build_dir, sources_dir). Tempdir if build_dir is None."""
        if build_dir is None:
            build_dir = Path(
                tempfile.mkdtemp(prefix=f"{self._build_dir_name}-{solution.name[:32]}-")
            )
        else:
            build_dir.mkdir(parents=True, exist_ok=True)
        sources_dir = build_dir / "sources"
        sources_dir.mkdir(parents=True, exist_ok=True)
        return build_dir, sources_dir

    def _materialize_sources(self, solution: Solution, sources_dir: Path) -> List[Path]:
        """Write solution.sources to disk; return the compilable .cpp paths."""
        src_paths: List[Path] = []
        for src in solution.sources:
            dst = sources_dir / src.path
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(src.content)
            if dst.suffix in (".cpp", ".cc", ".cxx"):
                src_paths.append(dst)
        return src_paths

    def _run_clang(self, cmd: List[str], solution: Solution) -> None:
        """Run the compile command; raise CompileError on non-zero exit."""
        logger.info("compile: %s", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise CompileError(
                f"clang++ failed (rc={proc.returncode}) for solution '{solution.name}'",
                returncode=proc.returncode,
                stderr=proc.stderr,
                command=cmd,
            )


__all__ = [
    "ARM_BENCH_ROOT",
    "BENCH_ROOT",
    "solutions_root",
    "Builder",
    "BuildError",
    "CompileError",
    "CompileResult",
]
