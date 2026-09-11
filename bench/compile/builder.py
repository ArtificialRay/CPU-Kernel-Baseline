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

import functools
import logging
import os
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from bench.data.definition import Definition
from bench.data.solution import Solution

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _resolve_cxx() -> Optional[str]:
    """Return the first available clang++ binary on this host."""
    for name in ("clang++", "clang++-18", "clang++-17", "clang++-16"):
        if shutil.which(name):
            return name
    return None


# ── Build error ──────────────────────────────────────────────────────────────

class CompileError(RuntimeError):
    """clang++ returned non-zero. The message includes the full stderr."""

    def __init__(self, message: str, returncode: int, stderr: str, command: List[str]):
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr
        self.command = command


# A stub used by the registry fo "no builder can build this" / "build failed" conditions.
# Alias matching flashinfer-bench's naming, 
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


# isa tier -> /proc/cpuinfo "Features" tokens that must all be present, highest
# tier first. Mirrors mcp_app/agent_tools/isa.py's _ISA_CPUINFO_TOKENS.
_NATIVE_TIERS = (("sve2", ("sve2",)), ("sve", ("sve",)), ("neon", ("asimd",)))


def resolve_native_march(flags: List[str]) -> List[str]:
    """Replace `-march=native` with a concrete `-march=` for this host.

    clang's AArch64 host detection doesn't know every Graviton part (clang-18
    on Graviton3, part 0xd40, resolves `-march=native` to `-target-cpu generic`
    with NO SVE), so a baseline that bakes `-march=native` — every simd-loop
    baseline does — compiles without SVE and fails, leaving evaluate() with no
    baseline and a null speedup. Pick the highest tier whose cpuinfo tokens
    are all present and use the same march candidates are compiled with
    (contracts.ISA_TABLE). ARMBENCH_NATIVE_MARCH overrides the choice."""
    if "-march=native" not in flags:
        return list(flags)
    march = os.environ.get("ARMBENCH_NATIVE_MARCH")
    if not march:
        try:
            from contracts import ISA_TABLE  # repo-root module, present on the box
            feats = ""
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("Features"):
                    feats = line.split(":", 1)[1]
                    break
            tokens = set(feats.split())
            march = next((ISA_TABLE[t].march for t, need in _NATIVE_TIERS
                          if all(n in tokens for n in need)), None)
        except Exception:  # noqa: BLE001 — fall through to clang's own detection
            march = None
    if not march:
        return list(flags)
    return [march if f == "-march=native" else f for f in flags]

class Builder(ABC):
    """Base class for all builders.

    Subclasses declare a `_build_dir_name` prefix (for tempdir naming) and
    implement is_available / can_build / build.
    """

    _build_dir_name: str

    def __init__(self, build_dir_name: str) -> None:
        self._build_dir_name = build_dir_name

    @staticmethod
    def is_available() -> bool:
        """True if a clang++ binary is present on the host."""
        return _resolve_cxx() is not None

    @property
    def _cxx(self) -> str:
        """The clang++ binary to use (versioned fallback if unversioned is absent)."""
        return _resolve_cxx() or "clang++"

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
    "Builder",
    "BuildError",
    "CompileError",
    "CompileResult",
]
