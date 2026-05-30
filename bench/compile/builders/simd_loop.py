"""SimdLoopBuilder — per-dataset baseline stub for the `simd-loop` dataset.

Registered so the registry is multi-dataset ready, but the simd-loop dataset
(the loop_* SIMD baselines) has not been migrated into bench/ yet, so build()
raises NotImplementedError.
"""

from __future__ import annotations

import shutil

from bench.data.definition import Definition
from bench.data.solution import Solution, SupportedDatasets

from ..builder import Builder, CompileResult


class SimdLoopBuilder(Builder):
    def __init__(self) -> None:
        super().__init__(build_dir_name="armbench-simdloop")

    @staticmethod
    def is_available() -> bool:
        # clang++-gated like the others, so it registers wherever a compiler exists.
        return shutil.which("clang++") is not None

    def can_build(self, solution: Solution, is_baseline: bool) -> bool:
        return is_baseline and solution.dataset == SupportedDatasets.SIMD_LOOP

    def build(self, definition: Definition, solution: Solution) -> CompileResult:
        raise NotImplementedError(
            "simd-loop builder lands when the loop_* dataset is migrated into bench/"
        )
