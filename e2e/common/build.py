"""Compiling a kernel into a shared object for a framework to dlopen, and checking it can be."""
from __future__ import annotations

import subprocess
from pathlib import Path


def compile_shared(cxx: str, march: str, include_dir: Path, sources: list, so: Path, *,
                   opt: str = "-O3", std: str = "c++14") -> subprocess.CompletedProcess:
    """`cxx -shared -fPIC` the sources into `so`, the way the evaluation harness compiled them
    (same optimisation level, standard and ISA flag)."""
    cmd = [cxx, "-shared", "-fPIC", opt, march, f"-std={std}", "-I", str(include_dir),
           *map(str, sources), "-o", str(so)]
    return subprocess.run(cmd, capture_output=True, text=True)


def undefined_symbols(so: Path, prefix: str) -> list[str]:
    """Symbols starting with `prefix` that the shared object expects someone else to provide.

    A kernel that calls the framework's own routine compiles and runs inside the evaluation
    harness (which links the framework) -- and so scores ~1.00x, being the baseline. Loaded
    on its own by the framework's override hook the symbol does not resolve, and the loader
    then disables every override, not just that kernel's. Catch it before it gets there."""
    try:
        out = subprocess.run(["nm", "-D", "-u", str(so)], capture_output=True, text=True).stdout
    except FileNotFoundError:
        return []
    return sorted({tok for line in out.splitlines() for tok in line.split() if tok.startswith(prefix)})
