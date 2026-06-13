"""CandidateBuilder — the shared raw-`float*` path for ALL candidate solutions.

Every non-baseline solution is built here, regardless of `solution.dataset`.
Candidates have NO ncnn dependency: the build is just the solution sources plus
(optionally) the raw harness forwarder, compiled with clang++ into a `.so`.

The C-ABI contract lives in `candidate_harness/<op_type>.h` (shipped with the
builder, not per-dataset). The matching ctypes argtypes live in
`bench/datasets/raw.py`.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

from bench.data.definition import Definition
from bench.data.solution import Solution

from ..builder import Builder, CompileError, CompileResult

# candidate_harness/ ships next to this file.
_HARNESS_DIR = Path(__file__).resolve().parent / "candidate_harness"


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
        op_type = definition.op_type
        harness_h = _HARNESS_DIR / f"{op_type}.h"
        harness_cpp = _HARNESS_DIR / f"{op_type}.cpp"
        if not harness_h.exists():
            raise FileNotFoundError(
                f"Candidate harness header missing: {harness_h}. The raw `float*` "
                f"contract for op_type '{op_type}' has not been ported yet."
            )

        build_dir, sources_dir = self._make_build_dir(solution)
        solution_src_paths = self._materialize_sources(solution, sources_dir)

        # If the kernel exports the C-ABI entry itself, compiling the forwarder
        # would duplicate-define armbench_entry_<op>. Only add the forwarder when
        # the author used the alternate `armbench_<op>_kernel` style.
        entry_symbol = solution.get_entry_symbol()
        compile_harness = entry_symbol != f"armbench_entry_{op_type}"

        so_path = build_dir / f"{solution.name[:64]}.so"
        cmd: List[str] = ["clang++", "-shared", "-fPIC"]
        cmd += list(solution.spec.compile_flags or [])

        # Include dirs: only the solution's own sources. 
        cmd += ["-I", str(sources_dir)]

        if compile_harness:
            if not harness_cpp.exists():
                raise FileNotFoundError(
                    f"Candidate '{solution.name}' uses entry symbol '{entry_symbol}', "
                    f"which requires the forwarder {harness_cpp}, but it is missing."
                )
            cmd.append(str(harness_cpp))
        cmd += [str(p) for p in solution_src_paths]
        cmd += ["-o", str(so_path)]
        cmd += list(solution.spec.link_flags or [])

        try:
            self._run_clang(cmd, solution)
        except CompileError:
            shutil.rmtree(build_dir, ignore_errors=True)
            raise

        return CompileResult(so_path=so_path, build_dir=build_dir, command=cmd)
