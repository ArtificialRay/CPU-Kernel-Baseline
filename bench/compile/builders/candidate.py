"""CandidateBuilder — the shared raw-`float*` path for every solution no
framework-specific builder wants: all candidates, plus any dataset whose true
baseline also has no framework dependency of its own (e.g. kleidiai — see
SupportedDatasets.KLEIDIAI).

CandidateBuilder is for solution where each solution embeds its own binding in its sources (armbench_entry_<op> is
defined in the solution's own files) 
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

    def can_build(self, solution: Solution, is_baseline: bool) -> bool:
        """Unconditional fallback — see registry.py's _BUILDER_PRIORITY: this
        builder is tried last, after every dataset with its own
        framework-specific builder (ncnn's true baseline, simd-loop,
        llama.cpp) has already had first refusal and declined. No dataset
        enumeration needed here — a new no-framework dataset is picked up
        automatically, with zero changes to this file."""
        return True

    def build(self, definition: Definition, solution: Solution) -> CompileResult:
        build_dir, sources_dir = self._make_build_dir(solution)
        solution_src_paths = self._materialize_sources(solution, sources_dir)

        so_path = build_dir / f"{solution.name[:64]}.so"
        cmd: List[str] = [self._cxx, "-shared", "-fPIC"]
        cmd += list(solution.spec.compile_flags or [])

        # Include dirs: only the solution's own sources.
        cmd += ["-I", str(sources_dir)]

        cmd += self._source_compile_args(solution_src_paths)
        cmd += ["-o", str(so_path)]
        cmd += list(solution.spec.link_flags or [])

        try:
            self._run_clang(cmd, solution)
        except CompileError:
            shutil.rmtree(build_dir, ignore_errors=True)
            raise

        return CompileResult(so_path=so_path, build_dir=build_dir, command=cmd)
