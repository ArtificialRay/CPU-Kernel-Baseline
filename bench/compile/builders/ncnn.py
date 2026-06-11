"""NcnnBuilder — baseline path for the `ncnn` dataset.

Builds, against the real ncnn checkout (`ncnn/src/*.cpp`, `ncnn/src/layer/arm/`):

  - the solution's own sources — kernel.cpp (delegates to the ncnn arm layer)
    plus binding.cpp defining armbench_entry_<op> (the ncnn::Mat/Option shim,
    with per-Definition params baked as constexpr) and the <op>.h contract
  - bench/datasets/_ncnn_lib/_mat_factory.cpp (numpy ↔ ncnn::Mat)

into one .so (linked against the full ncnn static lib) that the runner dlopens.

Every baseline Solution is self-contained: it ships its own binding.cpp, so the
builder injects no shared op harness — it just compiles the Solution's sources.

Host setup (Graviton / AArch64 Ubuntu):
    sudo apt-get install clang-18 libomp-18-dev cmake
    git clone --depth=1 https://github.com/Tencent/ncnn.git ncnn
    cd ncnn && cmake -B build -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_TESTS=OFF \\
        -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF \\
        -DNCNN_VULKAN=OFF -DNCNN_SHARED_LIB=OFF \\
        -DCMAKE_C_COMPILER=clang-18 -DCMAKE_CXX_COMPILER=clang++-18
    cmake --build build -j$(nproc) ncnn   # → ncnn/build/src/libncnn.a

The real ncnn kernels come from a full `libncnn.a` produced by ncnn's own CMake
build (see `_ncnn_static_lib`). That build is required: upstream
`convolution_arm.cpp` references a sprawling tree of per-ISA helper translation
units (`convolution_arm_i8mm.cpp`, `..._asimddp.cpp`, `..._sve.cpp`, …), each
compiled by ncnn's CMake with its own `-march` flags — hand-picking a subset
into a slim archive leaves undefined symbols at dlopen. The same CMake build also
generates `platform.h` under `<base_root>/build/src`, which we add to the include
path.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Optional

from bench.data.definition import Definition
from bench.data.solution import Solution, SupportedDatasets

from ..builder import (
    Builder,
    CompileError,
    CompileResult,
)


def _ncnn_base_root(base_root: Path) -> Path:
    """ncnn checkout root. Lives inside the repo at <repo_root>/ncnn/.

    By default ncnn/ is a direct child of the repo root (arm_bench_root).
    Overridable via NCNN_ROOT (points at the ncnn checkout, e.g.
    /home/ubuntu/ncnn).
    """
    env = os.environ.get("NCNN_ROOT")
    if env:
        return Path(env).expanduser()
    return base_root / "ncnn"


def _ncnn_static_lib(base_root: Path) -> Optional[Path]:
    """Locate the full ncnn static lib from a CMake build, if one exists.

    A real `libncnn.a` resolves Convolution_arm's entire dependency tree (every
    per-ISA helper TU) and ships the generated platform.h next to it. Build it
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
        self._arm_bench_root = Path(__file__).resolve().parents[3] # repo dir
        self._base_root = self._arm_bench_root.parent # home dir
    @staticmethod
    def is_available() -> bool:
        return shutil.which("clang++") is not None

    def can_build(self, solution: Solution, is_baseline: bool) -> bool:
        return is_baseline and solution.dataset == SupportedDatasets.NCNN

    def build(self, definition: Definition, solution: Solution) -> CompileResult:
        arm_bench_root = self._arm_bench_root
        base_root = _ncnn_base_root(self._base_root)

        build_dir, sources_dir = self._make_build_dir(solution)
        try:
            return self._build_inner(
                solution, arm_bench_root, base_root,
                build_dir, sources_dir,
            )
        except (CompileError, FileNotFoundError):
            shutil.rmtree(build_dir, ignore_errors=True)
            raise

    def _build_inner(
        self, solution, arm_bench_root, base_root,
        build_dir, sources_dir,
    ) -> CompileResult:
        solution_src_paths = self._materialize_sources(solution, sources_dir)

        # Every baseline Solution include a kernel source (kernel.cpp), 
        # binding.cpp defining armbench_entry_<op> (the ncnn::Mat shim, with
        # per-Definition params baked as constexpr) plus the kernel it calls.
        # The builder just compiles those sources.
        mat_factory_cpp = (
            arm_bench_root / "bench" / "datasets" / "_ncnn_lib" / "_mat_factory.cpp"
        )
        if not mat_factory_cpp.exists():
            raise FileNotFoundError(f"ncnn helper source not found: {mat_factory_cpp}")

        # Include dirs — retargeted to ncnn/src and ncnn/src/layer/arm.
        include_dirs = [
            sources_dir,
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

        # Link against the full ncnn static lib — it resolves the whole ncnn
        # dependency tree (Convolution_arm and its per-ISA helper TUs,
        # copy_make_border, Mat, ...) that the kernel and binding reference.
        static_lib = _ncnn_static_lib(base_root)
        if static_lib is None:
            raise FileNotFoundError(
                f"Full ncnn static lib not found under {base_root}/build "
                f"(build/src/libncnn.a or build/install/lib/libncnn.a). Build it "
                f"once with ncnn's CMake — see _ncnn_static_lib for the command — "
                f"or set NCNN_ROOT to a checkout that has one."
            )
        cmd.append(str(mat_factory_cpp))
        cmd += [str(p) for p in solution_src_paths]
        cmd += ["-o", str(so_path)]
        cmd.append(str(static_lib))

        cmd += list(solution.spec.link_flags or [])
        if "-fopenmp" not in cmd and not any("-fno-openmp" in f for f in cmd):
            cmd.append("-fopenmp")

        self._run_clang(cmd, solution)
        return CompileResult(so_path=so_path, build_dir=build_dir, command=cmd)
