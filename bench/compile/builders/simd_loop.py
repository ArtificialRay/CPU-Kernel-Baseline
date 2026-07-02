"""SimdLoopBuilder — baseline builder for the `simd-loop` dataset.

Compiles a SIMD-loop baseline kernel (e.g. the scalar or NEON reference for
loop_001) into a .so the runner can dlopen. No ncnn framework dependency — just
the solution's own sources. Like the ncnn baseline (which ships its own
binding.cpp), every simd-loop solution is self-contained: it carries its harness
shim fused into `solution.sources`, so the builder injects nothing from disk.

Fused-harness convention (solution.sources carries <op_type>.{h,cpp}):
  - The .h declares `struct loop_NNN_data` and
    `extern "C" int armbench_entry_loop_NNN(void*, void*, int64_t, void*)`.
  - The .cpp defines `armbench_entry_loop_NNN`, which fills the struct and calls
    `inner_loop_NNN(&data)` — the symbol the solution's kernel source exports.
  - The solution must define `inner_loop_NNN` as `extern "C"` (not static).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

from bench.data.definition import Definition
from bench.data.solution import Solution, SupportedDatasets

from ..builder import Builder, CompileError, CompileResult


class SimdLoopBuilder(Builder):
    def __init__(self) -> None:
        super().__init__(build_dir_name="armbench-simdloop")

    def can_build(self, solution: Solution, is_baseline: bool) -> bool:
        return solution.dataset == SupportedDatasets.SIMD_LOOP

    def build(self, definition: Definition, solution: Solution) -> CompileResult:
        op_type = definition.op_type
        build_dir, sources_dir = self._make_build_dir(solution)
        try:
            # The harness is fused into the solution sources (loop_NNN.h /
            # loop_NNN.cpp), the same way the ncnn baseline ships its own
            # binding.cpp. The builder just compiles the solution's own sources.
            harness_h_name   = f"{op_type}.h"
            harness_cpp_name = f"{op_type}.cpp"
            fused_harness_h   = next((s for s in solution.sources if s.path == harness_h_name), None)
            fused_harness_cpp = next((s for s in solution.sources if s.path == harness_cpp_name), None)
            if not (fused_harness_h and fused_harness_cpp):
                raise FileNotFoundError(
                    f"simd-loop solution '{solution.name}' is missing its fused harness "
                    f"sources ({harness_h_name} + {harness_cpp_name}). "
                    f"Re-run scripts/gen_simd_loop_harness.py to regenerate the solution."
                )

            solution_src_paths = self._materialize_sources(solution, sources_dir)
            include_dirs: List[Path] = [sources_dir]
            harness_cpp_path = sources_dir / harness_cpp_name

            so_path = build_dir / f"{solution.name[:64]}.so"
            cmd: List[str] = [self._cxx, "-shared", "-fPIC"]
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
