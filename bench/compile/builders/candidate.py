"""CandidateBuilder — the shared raw-`float*` path for ALL candidate solutions.

Every non-baseline solution is built here, regardless of `solution.dataset`.
Candidates have NO ncnn dependency: each solution embeds its own binding in
its sources (armbench_entry_<op> is defined in the solution's own files).
"""

from __future__ import annotations

import shutil
from typing import List

from bench.data.definition import Definition
from bench.data.solution import Solution

from ..builder import Builder, CompileError, CompileResult


class CandidateBuilder(Builder):
    """Builds candidate (non-baseline) kernels against the raw `float*` ABI."""

    def __init__(self) -> None:
        super().__init__(build_dir_name="armbench-cand")

    @staticmethod
    def is_available() -> bool:
        return shutil.which("clang++") is not None

    def can_build(self, solution: Solution, is_baseline: bool) -> bool:
        from bench.data.solution import SupportedDatasets
        # simd-loop solutions use SimdLoopBuilder regardless of is_baseline.
        return not is_baseline and solution.dataset != SupportedDatasets.SIMD_LOOP

    def build(self, definition: Definition, solution: Solution) -> CompileResult:
        build_dir, sources_dir = self._make_build_dir(solution)
        solution_src_paths = self._materialize_sources(solution, sources_dir)

        so_path = build_dir / f"{solution.name[:64]}.so"
        cmd: List[str] = ["clang++", "-shared", "-fPIC"]
        cmd += list(solution.spec.compile_flags or [])

        # Include dirs: only the solution's own sources.
        cmd += ["-I", str(sources_dir)]

        cmd += [str(p) for p in solution_src_paths]
        cmd += ["-o", str(so_path)]
        cmd += list(solution.spec.link_flags or [])

        try:
            self._run_clang(cmd, solution)
        except CompileError:
            shutil.rmtree(build_dir, ignore_errors=True)
            raise

        return CompileResult(so_path=so_path, build_dir=build_dir, command=cmd)
