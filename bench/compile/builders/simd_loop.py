"""SimdLoopBuilder — baseline builder for the `simd-loop` dataset.

Compiles a SIMD-loop baseline kernel (e.g. the scalar or NEON reference for
loop_001) into a .so the runner can dlopen. No ncnn framework dependency —
just the solution sources plus the per-op harness shim from simd_loop_harness/.

Harness convention (simd_loop_harness/<op_type>.{h,cpp}):
  - The .h declares `struct loop_NNN_data` and
    `extern "C" int armbench_entry_loop_NNN(void*, void*, int64_t, void*)`.
  - The .cpp defines `armbench_entry_loop_NNN`, which fills the struct and calls
    `inner_loop_NNN(&data)` — the symbol the solution source must export.
  - The solution must define `inner_loop_NNN` as `extern "C"` (not static).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

from bench.data.definition import Definition
from bench.data.solution import Solution, SupportedDatasets

from ..builder import Builder, CompileError, CompileResult

_HARNESS_DIR = Path(__file__).resolve().parent / "simd_loop_harness"


class SimdLoopBuilder(Builder):
    def __init__(self) -> None:
        super().__init__(build_dir_name="armbench-simdloop")

    @staticmethod
    def is_available() -> bool:
        return shutil.which("clang++") is not None

    def can_build(self, solution: Solution, is_baseline: bool) -> bool:
        return solution.dataset == SupportedDatasets.SIMD_LOOP

    def build(self, definition: Definition, solution: Solution) -> CompileResult:
        op_type = definition.op_type
        build_dir, sources_dir = self._make_build_dir(solution)
        try:
            # Prefer harness fused into solution sources (loop_NNN.h / loop_NNN.cpp).
            # Fall back to on-disk simd_loop_harness/ for solutions generated before
            # the fuse migration.
            harness_h_name   = f"{op_type}.h"
            harness_cpp_name = f"{op_type}.cpp"
            fused_harness_h   = next((s for s in solution.sources if s.path == harness_h_name), None)
            fused_harness_cpp = next((s for s in solution.sources if s.path == harness_cpp_name), None)

            if fused_harness_h and fused_harness_cpp:
                # Both harness files are embedded in the solution — materialize all.
                solution_src_paths = self._materialize_sources(solution, sources_dir)
                include_dirs: List[Path] = [sources_dir]
                harness_cpp_path = sources_dir / harness_cpp_name
            else:
                # Legacy path: read harness from simd_loop_harness/.
                harness_cpp_path = _HARNESS_DIR / f"{op_type}.cpp"
                harness_h_path   = _HARNESS_DIR / f"{op_type}.h"
                if not harness_h_path.exists():
                    raise FileNotFoundError(
                        f"simd-loop harness header missing: {harness_h_path}. "
                        f"Re-run scripts/gen_simd_loop_harness.py to regenerate."
                    )
                if not harness_cpp_path.exists():
                    raise FileNotFoundError(f"simd-loop harness shim missing: {harness_cpp_path}")
                solution_src_paths = self._materialize_sources(solution, sources_dir)
                include_dirs = [_HARNESS_DIR, sources_dir]

            so_path = build_dir / f"{solution.name[:64]}.so"
            cmd: List[str] = ["clang++", "-shared", "-fPIC"]
            cmd += list(solution.spec.compile_flags or [])
            for inc in include_dirs:
                cmd += ["-I", str(inc)]
            # Harness cpp first so its forward-decl of inner_loop_NNN is visible.
            cmd.append(str(harness_cpp_path))
            # Add user kernel sources (skip harness files already added above).
            harness_names = {harness_h_name, harness_cpp_name}
            cmd += [str(p) for p in solution_src_paths if p.name not in harness_names]
            cmd += list(solution.spec.link_flags or [])
            cmd += ["-o", str(so_path)]

            self._run_clang(cmd, solution)
            return CompileResult(so_path=so_path, build_dir=build_dir, command=cmd)
        except (CompileError, FileNotFoundError):
            shutil.rmtree(build_dir, ignore_errors=True)
            raise
