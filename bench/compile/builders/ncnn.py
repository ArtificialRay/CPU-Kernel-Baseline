"""NcnnBuilder — baseline path for the `ncnn` dataset.

Port of the old `compile_solution` (baseline branch only), retargeted from the
stale `framework/` + `arm-heavy-optimized/` layout to the real ncnn checkout
(`ncnn/src/*.cpp`, `ncnn/src/layer/arm/`). Builds:

  - the solution kernel.cpp (defines ncnn::convolution_kernel)
  - solutions/ncnn/_harness/<op>.{cpp,h}     (ncnn::Mat/Option shim)
  - bench/datasets/_ncnn_lib/_mat_factory.cpp (numpy ↔ ncnn::Mat)
  - bench/datasets/_ncnn_lib/_ncnn_*_stubs.cpp
  - the minimum ncnn framework sources

into one .so the runner dlopens.

ncnn needs a CMake-generated `platform.h`. We add `<base_root>/build/src` to the
include path when present (the conventional `cmake -B build` output), and the
prebuilt `libncnn.a` / `libncnn_arm_heavy.a` archive when the solution declares
the `ncnn_arm_heavy` dependency. Both are resolved at build time and the exact
shape is confirmed on the ARM host.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Optional

from bench.data.definition import Definition
from bench.data.solution import Solution, SupportedDatasets

from ..builder import (
    BENCH_ROOT,
    Builder,
    CompileError,
    CompileResult,
    solutions_root,
)

# Minimum ncnn framework .cpp set, retargeted to ncnn/src/ (was framework/*.cpp).
NCNN_FRAMEWORK_SOURCES = [
    "src/mat.cpp",
    "src/allocator.cpp",
    "src/expression.cpp",
    "src/option.cpp",
    "src/paramdict.cpp",
    "src/cpu.cpp",
    "src/modelbin.cpp",
    "src/datareader.cpp",
]


def _ncnn_base_root(arm_bench_root: Path) -> Path:
    """ncnn checkout root. Repo-root sibling of arm-bench by default.

    Overridable via ARMBENCH_BASE_ROOT for remote layouts.
    """
    env = os.environ.get("ARMBENCH_BASE_ROOT")
    if env:
        return Path(env).expanduser()
    return arm_bench_root.parent / "ncnn"


def _has_ncnn_arm_heavy(solution: Solution) -> bool:
    return "ncnn_arm_heavy" in (solution.spec.dependencies or [])


def _ncnn_arm_heavy_archive(arm_bench_root: Path) -> Path:
    return arm_bench_root / "build" / "libncnn_arm_heavy.a"


def _ncnn_static_lib(base_root: Path) -> Optional[Path]:
    """Locate a full ncnn static lib from a CMake build, if one exists.

    A real `libncnn.a` resolves Convolution_arm's entire dependency tree (and
    ships the generated platform.h next to it), so when present it supersedes
    the framework-sources + stubs + arm-heavy-archive path entirely. Build it
    once on the host with:

        cd ncnn && cmake -B build -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_TESTS=OFF \\
            -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF -DNCNN_VULKAN=OFF \\
            -DNCNN_SHARED_LIB=OFF && cmake --build build -j ncnn
    """
    for rel in ("build/src/libncnn.a", "build/install/lib/libncnn.a"):
        p = base_root / rel
        if p.exists():
            return p
    return None


class NcnnBuilder(Builder):
    """Builds the ncnn baseline solution against the real ncnn checkout."""

    def __init__(self) -> None:
        super().__init__(build_dir_name="armbench-ncnn")
        self._arm_bench_root = BENCH_ROOT.parent

    @staticmethod
    def is_available() -> bool:
        return shutil.which("clang++") is not None

    def can_build(self, solution: Solution, is_baseline: bool) -> bool:
        return is_baseline and solution.dataset == SupportedDatasets.NCNN

    def build(self, definition: Definition, solution: Solution) -> CompileResult:
        op_type = definition.op_type
        arm_bench_root = self._arm_bench_root
        base_root = _ncnn_base_root(arm_bench_root)

        build_dir, sources_dir = self._make_build_dir(solution)
        try:
            return self._build_inner(
                definition, solution, op_type, arm_bench_root, base_root,
                build_dir, sources_dir,
            )
        except (CompileError, FileNotFoundError):
            shutil.rmtree(build_dir, ignore_errors=True)
            raise

    def _build_inner(
        self, definition, solution, op_type, arm_bench_root, base_root,
        build_dir, sources_dir,
    ) -> CompileResult:
        solution_src_paths = self._materialize_sources(solution, sources_dir)

        # Harness (ncnn::Mat shim).
        harness_cpp = (
            solutions_root(arm_bench_root)
            / solution.dataset.value / "_harness" / f"{op_type}.cpp"
        )
        harness_h = harness_cpp.with_suffix(".h")
        if not harness_cpp.exists():
            raise FileNotFoundError(f"Dataset harness not found: {harness_cpp}")
        if not harness_h.exists():
            raise FileNotFoundError(f"Harness header not found: {harness_h}")

        mat_factory_cpp = BENCH_ROOT / "datasets" / "_ncnn_lib" / "_mat_factory.cpp"
        if not mat_factory_cpp.exists():
            raise FileNotFoundError(f"ncnn helper source not found: {mat_factory_cpp}")

        # Include dirs — retargeted to ncnn/src and ncnn/src/layer/arm.
        include_dirs = [
            sources_dir,
            harness_cpp.parent,
            base_root / "src",
            base_root / "src" / "layer",
            base_root / "src" / "layer" / "arm",
            arm_bench_root / "starter" / "ncnn",
        ]
        cmake_src = base_root / "build" / "src"
        if (cmake_src / "platform.h").exists():
            include_dirs.insert(0, cmake_src)

        so_path = build_dir / f"{solution.name[:64]}.so"
        cmd: List[str] = ["clang++", "-shared", "-fPIC"]
        cmd += list(solution.spec.compile_flags or [])
        if not any(f.startswith("-O") for f in cmd):
            cmd.append("-O2")
        if not any(f.startswith("-std=") for f in cmd):
            cmd.append("-std=c++14")
        for inc in include_dirs:
            cmd += ["-I", str(inc)]

        static_lib = _ncnn_static_lib(base_root)
        if static_lib is not None:
            # Full-lib path: libncnn.a resolves the whole ncnn dependency tree
            # (Convolution_arm, copy_make_border, Mat, ...). No framework sources,
            # no create_layer stubs — linking those too would duplicate symbols.
            cmd.append(str(harness_cpp))
            cmd.append(str(mat_factory_cpp))
            cmd += [str(p) for p in solution_src_paths]
            cmd += ["-o", str(so_path)]
            cmd.append(str(static_lib))
        else:
            # Legacy minimal path: hand-pick framework sources + create_layer
            # stubs + the optional prebuilt arm-heavy archive. Used when no full
            # ncnn CMake build is present.
            if _has_ncnn_arm_heavy(solution):
                stubs = BENCH_ROOT / "datasets" / "_ncnn_lib" / "_ncnn_arm_heavy_stubs.cpp"
            else:
                stubs = BENCH_ROOT / "datasets" / "_ncnn_lib" / "_ncnn_unused_stubs.cpp"
            if not stubs.exists():
                raise FileNotFoundError(f"ncnn helper source not found: {stubs}")
            framework_srcs: List[Path] = []
            for rel in NCNN_FRAMEWORK_SOURCES:
                p = base_root / rel
                if not p.exists():
                    raise FileNotFoundError(
                        f"ncnn framework source missing: {p}. Build a full ncnn "
                        f"static lib (see _ncnn_static_lib) or set ARMBENCH_BASE_ROOT."
                    )
                framework_srcs.append(p)
            heavy_archive: Optional[Path] = None
            if _has_ncnn_arm_heavy(solution):
                heavy_archive = _ncnn_arm_heavy_archive(arm_bench_root)
                if not heavy_archive.exists():
                    raise FileNotFoundError(
                        f"Solution '{solution.name}' depends on ncnn_arm_heavy but neither "
                        f"a full ncnn static lib nor the prebuilt archive {heavy_archive} "
                        f"is present."
                    )
            cmd.append(str(harness_cpp))
            cmd.append(str(mat_factory_cpp))
            cmd.append(str(stubs))
            cmd += [str(p) for p in framework_srcs]
            cmd += [str(p) for p in solution_src_paths]
            cmd += ["-o", str(so_path)]
            if heavy_archive is not None:
                cmd.append(str(heavy_archive))

        cmd += list(solution.spec.link_flags or [])
        if "-fopenmp" not in cmd and not any("-fno-openmp" in f for f in cmd):
            cmd.append("-fopenmp")

        self._run_clang(cmd, solution)
        return CompileResult(so_path=so_path, build_dir=build_dir, command=cmd)
