"""Compile a Solution into a .so for dlopen.

Pulls together:
  - solution.sources                          (the kernel author wrote this)
  - solutions/<dataset>/_harness/<op>.{cpp,h} (dataset's shared harness)
  - bench/datasets/_ncnn_lib/_mat_factory.cpp (numpy ↔ ncnn::Mat bridge,
                                                only for `dataset=ncnn`)
  - minimum ncnn framework sources            (from $BASE_ROOT/ncnn/framework/)

All compiled with clang++ + solution.spec.compile_flags + link_flags into one
shared library. The runner dlopens that .so and resolves both the
armbench_entry_<op_type> and armbench_ncnn_* symbols from it.

Build is self-contained — no CMake, no cached .a hunting. Rebuilds the
framework objects on every compile (a few hundred ms; cache on disk later
if it ever matters).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from bench.data.solution import Solution, SupportedDatasets

logger = logging.getLogger(__name__)


# ── Paths ────────────────────────────────────────────────────────────────────

# `arm-bench/`. We resolve relative to this file's location.
ARM_BENCH_ROOT = Path(__file__).resolve().parent.parent

# `bench/` package root.
BENCH_ROOT = Path(__file__).resolve().parent


def _solutions_root(arm_bench_root: Path) -> Path:
    # Solutions live inside the warehouse root at arm-bench/bench-trace/.
    # The `_harness/<op>.{cpp,h}` files are resolved off this path.
    return arm_bench_root / "bench-trace" / "solutions"


def _ncnn_base_root(arm_bench_root: Path) -> Path:
    """Path to the ncnn baseline checkout (sibling of arm-bench by default).

    Mirrors the CMakeLists.txt convention BASE_ROOT=${ARM_BENCH}/../ncnn.
    Overridable via env var ARMBENCH_BASE_ROOT for remote layouts.
    """
    env = os.environ.get("ARMBENCH_BASE_ROOT")
    if env:
        return Path(env).expanduser()
    return arm_bench_root.parent / "ncnn"


# ── Build error ──────────────────────────────────────────────────────────────

class CompileError(RuntimeError):
    """clang++ returned non-zero. The error message includes the full stderr."""

    def __init__(self, message: str, returncode: int, stderr: str, command: List[str]):
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr
        self.command = command


# ── Compile result ───────────────────────────────────────────────────────────

@dataclass
class CompileResult:
    so_path: Path
    """Absolute path to the produced .so."""
    build_dir: Path
    """Temp dir holding the build artifacts (sources, .o, .so). Caller owns cleanup
    via shutil.rmtree(build_dir) when done with the .so."""
    command: List[str]
    """The clang++ command that was run."""


# ── ncnn framework source list ───────────────────────────────────────────────

# Minimum set of ncnn .cpp files we need to link to satisfy our harness.
#   - mat.cpp        → Mat class + copy_make_border (used by harness padding)
#   - allocator.cpp  → Allocator base + dtor
#   - expression.cpp → used by Mat for shape computations
#   - option.cpp     → Option ctor + default field values
#   - layer.cpp      → Layer base + create_pipeline/destroy_pipeline defaults
# Headers only (no .cpp needed): ncnn_helpers.h, ref_conv.h.
NCNN_FRAMEWORK_SOURCES = [
    "framework/mat.cpp",
    "framework/allocator.cpp",
    "framework/expression.cpp",
    "framework/option.cpp",
    "framework/paramdict.cpp",
    "framework/cpu.cpp",
    "framework/modelbin.cpp",
    "datareader.cpp",  # NB: lives at ncnn root, not under framework/
]


# ── Compile entry point ──────────────────────────────────────────────────────

def _has_ncnn_arm_heavy(solution: Solution) -> bool:
    """True if the Solution declares the prebuilt arm-heavy lib as a dependency."""
    return "ncnn_arm_heavy" in (solution.spec.dependencies or [])


def _ncnn_arm_heavy_archive(arm_bench_root: Path) -> Path:
    """Path to the prebuilt static archive produced by scripts/build_ncnn_arm_heavy.sh."""
    return arm_bench_root / "build" / "libncnn_arm_heavy.a"


def compile_solution(
    solution: Solution,
    *,
    op_type: str,
    arm_bench_root: Optional[Path] = None,
    extra_include_dirs: Optional[List[Path]] = None,
    extra_objects: Optional[List[Path]] = None,
    compiler: str = "clang++",
    build_dir: Optional[Path] = None,
) -> CompileResult:
    """Compile `solution` + its dataset harness + ncnn framework into one .so.

    Parameters
    ----------
    solution
        The Solution to compile. `solution.dataset` picks the harness/helper set.
    op_type
        Derived from the Definition's op_type. Drives which `_harness/<op>.cpp`
        and SIGNATURES entry get used.
    arm_bench_root
        Defaults to the inferred path of arm-bench/ from this file.
    extra_include_dirs
        Extra -I dirs. Useful for tests pointing into the repo.
    extra_objects
        Pre-compiled .o files to link in. Reserved for cache use.
    compiler
        clang++ by default. Use g++ if you want.
    build_dir
        Where to materialize sources and emit .so. If None, a tempdir is created.
        Caller is responsible for cleaning up the returned `result.build_dir`.

    Returns
    -------
    CompileResult with the .so path.

    Raises
    ------
    CompileError if clang++ returns non-zero.
    FileNotFoundError if the harness or ncnn sources are missing on disk.
    """
    arm_bench_root = arm_bench_root or ARM_BENCH_ROOT
    base_root = _ncnn_base_root(arm_bench_root)

    if solution.dataset != SupportedDatasets.NCNN:
        raise NotImplementedError(
            f"compile.py: dataset '{solution.dataset.value}' not yet supported "
            f"(only 'ncnn' for Phase 1)"
        )

    # 1. Build dir
    if build_dir is None:
        build_dir = Path(tempfile.mkdtemp(prefix=f"armbench-{solution.name[:32]}-"))
    else:
        build_dir.mkdir(parents=True, exist_ok=True)
    sources_dir = build_dir / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)

    # 2. Materialize solution sources
    solution_src_paths: List[Path] = []
    for src in solution.sources:
        dst = sources_dir / src.path
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(src.content)
        if dst.suffix in (".cpp", ".cc", ".cxx"):
            solution_src_paths.append(dst)

    # 3. Resolve harness path
    harness_cpp = (
        _solutions_root(arm_bench_root) / solution.dataset.value / "_harness" / f"{op_type}.cpp"
    )
    harness_h = harness_cpp.with_suffix(".h")
    if not harness_cpp.exists():
        raise FileNotFoundError(
            f"Dataset harness not found: {harness_cpp}. Did you forget to add "
            f"solutions/{solution.dataset.value}/_harness/{op_type}.cpp?"
        )
    if not harness_h.exists():
        raise FileNotFoundError(f"Harness header not found: {harness_h}")

    # 4. mat_factory (dataset Python adapter's C side) + stubs for unreached
    # ncnn symbols (Layer factory, Layer base) that mat.cpp pulls in
    # transitively via copy_make_border.
    #
    # If the Solution declares ncnn_arm_heavy as a dependency, swap the
    # abort-on-call stubs for the heavy-build stubs that route create_layer
    # to real Padding / Convolution_arm layers (PHASE2.md deliverable #2).
    mat_factory_cpp = BENCH_ROOT / "datasets" / "_ncnn_lib" / "_mat_factory.cpp"
    if _has_ncnn_arm_heavy(solution):
        ncnn_stubs_cpp = BENCH_ROOT / "datasets" / "_ncnn_lib" / "_ncnn_arm_heavy_stubs.cpp"
    else:
        ncnn_stubs_cpp = BENCH_ROOT / "datasets" / "_ncnn_lib" / "_ncnn_unused_stubs.cpp"
    if not mat_factory_cpp.exists():
        raise FileNotFoundError(f"_mat_factory.cpp not found at {mat_factory_cpp}")
    if not ncnn_stubs_cpp.exists():
        raise FileNotFoundError(f"ncnn stubs file not found at {ncnn_stubs_cpp}")

    # 5. ncnn framework sources
    framework_srcs: List[Path] = []
    framework_rels = list(NCNN_FRAMEWORK_SOURCES)
    if _has_ncnn_arm_heavy(solution):
        # Padding layer impl — required by mat.cpp's copy_make_border path that
        # the harness hits whenever pad > 0. Without it, _make_layer would
        # return a Padding* whose vtable has no impl.
        framework_rels.append("c-partially-optimized/tensor/padding.cpp")
    for rel in framework_rels:
        p = base_root / rel
        if not p.exists():
            raise FileNotFoundError(
                f"ncnn framework source missing: {p}. Set ARMBENCH_BASE_ROOT "
                f"env var if your ncnn checkout is elsewhere."
            )
        framework_srcs.append(p)

    # 6. Include dirs
    include_dirs = [
        sources_dir,                              # solution's own headers
        harness_cpp.parent,                       # _harness/<op>.h
        base_root,                                # "framework/mat.h" etc.
        base_root / "framework",                  # unqualified "mat.h" / "option.h"
        arm_bench_root / "starter" / "ncnn",     # legacy ncnn_helpers.h (if a Solution still uses it)
    ]
    if _has_ncnn_arm_heavy(solution):
        # convolution_arm.h + convolution.h (base) + padding.h
        include_dirs += [
            base_root / "arm-heavy-optimized" / "conv",
            base_root / "arm-heavy-optimized" / "common",
            base_root / "c-partially-optimized" / "conv",
            base_root / "c-partially-optimized" / "tensor",
            base_root / "c-partially-optimized" / "common",
        ]
    if extra_include_dirs:
        include_dirs.extend(extra_include_dirs)

    # 6b. Prebuilt static archives (Phase 2: ncnn_arm_heavy)
    heavy_archive: Optional[Path] = None
    if _has_ncnn_arm_heavy(solution):
        heavy_archive = _ncnn_arm_heavy_archive(arm_bench_root)
        if not heavy_archive.exists():
            raise FileNotFoundError(
                f"Solution '{solution.name}' depends on ncnn_arm_heavy but the "
                f"prebuilt archive is missing: {heavy_archive}. "
                f"Run scripts/build_ncnn_arm_heavy.sh first."
            )

    # 7. Build the command
    so_path = build_dir / f"{solution.name[:64]}.so"

    cmd: List[str] = [compiler, "-shared", "-fPIC"]
    cmd += list(solution.spec.compile_flags or [])
    if not any(f.startswith("-O") for f in cmd):
        cmd.append("-O2")  # safe default
    if not any(f.startswith("-std=") for f in cmd):
        cmd.append("-std=c++14")

    for inc in include_dirs:
        cmd += ["-I", str(inc)]

    # Sources, in build order (harness first so any forward decls resolve)
    cmd.append(str(harness_cpp))
    cmd.append(str(mat_factory_cpp))
    cmd.append(str(ncnn_stubs_cpp))
    cmd += [str(p) for p in framework_srcs]
    cmd += [str(p) for p in solution_src_paths]

    # Extra pre-built objects
    if extra_objects:
        cmd += [str(o) for o in extra_objects]

    # Output
    cmd += ["-o", str(so_path)]

    # Prebuilt static archives — placed after sources, before -l flags, so
    # symbols satisfy the .o references seen so far. Use the full path (not
    # -L/-l) to avoid PATH-search ambiguity.
    if heavy_archive is not None:
        cmd.append(str(heavy_archive))

    # Link flags
    cmd += list(solution.spec.link_flags or [])
    # OpenMP is almost always required for ncnn-side code (allocator etc).
    # Add -fopenmp if not already present and the user hasn't disabled.
    if "-fopenmp" not in cmd and not any("-fno-openmp" in f for f in cmd):
        cmd.append("-fopenmp")

    logger.info("compile: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise CompileError(
            f"clang++ failed (rc={proc.returncode}) for solution '{solution.name}'",
            returncode=proc.returncode,
            stderr=proc.stderr,
            command=cmd,
        )

    return CompileResult(so_path=so_path, build_dir=build_dir, command=cmd)


def cleanup_build_dir(result: CompileResult) -> None:
    """Remove the build dir created by `compile_solution`. Idempotent."""
    if result.build_dir.exists():
        shutil.rmtree(result.build_dir, ignore_errors=True)


__all__ = [
    "ARM_BENCH_ROOT",
    "BENCH_ROOT",
    "CompileError",
    "CompileResult",
    "compile_solution",
    "cleanup_build_dir",
]
