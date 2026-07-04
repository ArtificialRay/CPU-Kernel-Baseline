"""LlamaCppBuilder — baseline path for the `llama.cpp` dataset.

Builds, against a real llama.cpp checkout's ggml static libs:

  - the solution's own sources — kernel.cpp (builds + computes the ggml graph)
    plus binding.cpp defining armbench_entry_<op> (unpacks the generic
    `const void* const*` input slots, with per-Definition axes baked as
    constexpr) and the <op>.h contract

into one .so (linked against libggml.a + libggml-cpu.a + libggml-base.a) that
the runner dlopens.

Every baseline Solution is self-contained: it ships its own binding.cpp, so the
builder injects no shared op harness — it just compiles the Solution's sources.

Host setup (macOS arm64 dev box or Graviton / AArch64 Ubuntu):
    # Graviton: sudo apt-get install clang-18 libomp-18-dev cmake
    git clone --depth=1 https://github.com/ggml-org/llama.cpp llama.cpp
    cd llama.cpp && cmake -B build \\
        -DGGML_METAL=OFF -DGGML_BLAS=OFF -DGGML_ACCELERATE=OFF \\
        -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON \\
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \\
        -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF
        # on Graviton add: -DCMAKE_C_COMPILER=clang-18 -DCMAKE_CXX_COMPILER=clang++-18
    cmake --build build -j --target ggml ggml-base ggml-cpu
        # → llama.cpp/build/ggml/src/libggml{,-base,-cpu}.a

GGML_METAL/BLAS/ACCELERATE are disabled so the link set is deterministic across
macOS and Graviton (pure ggml-cpu; GGML_NATIVE gives NEON on Apple Silicon and
SVE2 on Graviton4). Only the three ggml targets are needed — the full build
includes llama-app, which is irrelevant here.

The checkout root is resolved like NcnnBuilder resolves ncnn/ (env var first):
LLAMA_CPP_ROOT > <repo_root>/llama.cpp > <repo_root>/../llama.cpp.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional

from bench.data.definition import Definition
from bench.data.solution import Solution, SupportedDatasets

from ..builder import (
    Builder,
    CompileError,
    CompileResult,
)

_GGML_STATIC_LIBS = ("libggml.a", "libggml-cpu.a", "libggml-base.a")


def _llama_cpp_root(repo_root: Path) -> Path:
    """llama.cpp checkout root.

    Overridable via LLAMA_CPP_ROOT (points at the llama.cpp checkout, e.g.
    /home/ubuntu/llama.cpp). Defaults to <repo_root>/llama.cpp, falling back to
    a sibling of the repo (matching where the ncnn checkout convention allows
    either layout).
    """
    env = os.environ.get("LLAMA_CPP_ROOT")
    if env:
        return Path(env).expanduser()
    in_repo = repo_root / "llama.cpp"
    if in_repo.exists():
        return in_repo
    return repo_root.parent / "llama.cpp"


def _ggml_static_libs(base_root: Path) -> Optional[List[Path]]:
    """Locate the three ggml static libs from a CMake build, if they exist."""
    lib_dir = base_root / "build" / "ggml" / "src"
    libs = [lib_dir / name for name in _GGML_STATIC_LIBS]
    if all(p.exists() for p in libs):
        return libs
    return None


class LlamaCppBuilder(Builder):
    """Builds llama.cpp baseline solutions against the ggml static libs."""

    def __init__(self) -> None:
        super().__init__(build_dir_name="armbench-llamacpp")
        self._arm_bench_root = Path(__file__).resolve().parents[3]  # repo dir

    def can_build(self, solution: Solution, is_baseline: bool) -> bool:
        # llama.cpp solutions are inherently framework-bound baselines; like
        # simd-loop, dispatch purely on the dataset so `bench.cli bench` works
        # without a baseline_author match.
        return solution.dataset == SupportedDatasets.LLAMA_CPP

    def build(self, definition: Definition, solution: Solution) -> CompileResult:
        base_root = _llama_cpp_root(self._arm_bench_root)

        build_dir, sources_dir = self._make_build_dir(solution)
        try:
            return self._build_inner(solution, base_root, build_dir, sources_dir)
        except (CompileError, FileNotFoundError):
            shutil.rmtree(build_dir, ignore_errors=True)
            raise

    def _build_inner(
        self, solution, base_root, build_dir, sources_dir,
    ) -> CompileResult:
        solution_src_paths = self._materialize_sources(solution, sources_dir)

        static_libs = _ggml_static_libs(base_root)
        if static_libs is None:
            raise FileNotFoundError(
                f"ggml static libs not found under {base_root}/build/ggml/src "
                f"({', '.join(_GGML_STATIC_LIBS)}). Build them once with llama.cpp's "
                f"CMake — see the module docstring for the command — or set "
                f"LLAMA_CPP_ROOT to a checkout that has them."
            )

        include_dirs = [
            sources_dir,
            base_root / "ggml" / "include",
        ]

        so_path = build_dir / f"{solution.name[:64]}.so"
        cmd: List[str] = [self._cxx, "-shared", "-fPIC"]
        cmd += list(solution.spec.compile_flags or [])
        for inc in include_dirs:
            cmd += ["-I", str(inc)]

        cmd += [str(p) for p in solution_src_paths]
        cmd += ["-o", str(so_path)]
        # Link order: libggml (backend registry) → libggml-cpu → libggml-base.
        cmd += [str(p) for p in static_libs]

        if sys.platform == "linux":
            # ggml-cpu built per the docstring uses OpenMP + pthreads on Linux;
            # harmless if the local build disabled OpenMP.
            cmd += ["-fopenmp", "-lpthread", "-lm", "-ldl"]

        cmd += list(solution.spec.link_flags or [])

        self._run_clang(cmd, solution)
        return CompileResult(so_path=so_path, build_dir=build_dir, command=cmd)
