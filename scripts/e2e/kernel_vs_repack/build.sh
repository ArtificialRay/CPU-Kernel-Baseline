#!/usr/bin/env bash
# build.sh <llama.cpp dir> <llama.cpp build dir> [out binary]
# Compiles bench_repack.cpp against an EXISTING llama.cpp/ggml build (headers from <llama.cpp dir>/ggml/include,
# libggml{,-cpu,-base} from <build dir>, static .a or shared .so -- whichever the build produced).
#   e2e box:     scripts/e2e/kernel_vs_repack/build.sh ~/llama.cpp-e2e ~/llama.cpp-e2e/build-stock
# Link against build-stock (GGML_CPU_REPACK=ON, the default); a GGML_CPU_REPACK=OFF build has no CPU_REPACK buft.
# CXXFLAGS_ARCH overrides -mcpu=native (e.g. -march=armv8.2-a+dotprod for a compile-only check).
set -euo pipefail
L="${1:?llama.cpp source dir}"; B="${2:?llama.cpp build dir}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${3:-$HERE/bench_repack}"
CXX="${CXX:-clang++-18}"
ARCH="${CXXFLAGS_ARCH:--mcpu=native}"

find_lib() {  # prefer the static archive, else the shared object
  local n="$1" f
  for f in "$B/ggml/src/lib$n.a" "$B/ggml/src/lib$n.so" "$B/bin/lib$n.so" "$B/lib/lib$n.a" "$B/lib/lib$n.so"; do
    [ -e "$f" ] && { echo "$f"; return; }
  done
  f=$(find "$B" -name "lib$n.a" -o -name "lib$n.so" 2>/dev/null | head -1)
  [ -n "$f" ] && { echo "$f"; return; }
  echo "missing lib$n in $B (build it: cmake --build $B --target ggml ggml-base ggml-cpu)" >&2; exit 1
}
LG=$(find_lib ggml); LC=$(find_lib ggml-cpu); LB=$(find_lib ggml-base)
EXTRA=()
# the ggml-cpu archive needs OpenMP only if the build found it
if grep -q "^GGML_OPENMP_ENABLED:INTERNAL=ON" "$B/CMakeCache.txt" 2>/dev/null; then EXTRA+=(-fopenmp); fi
grep -q "^GGML_CPU_REPACK:BOOL=OFF" "$B/CMakeCache.txt" 2>/dev/null && \
  echo "WARNING: $B was built with GGML_CPU_REPACK=OFF -- repack timings will be null" >&2
RPATH=()
case "$LG" in *.so) RPATH=(-Wl,-rpath,"$(dirname "$LG")":"$(dirname "$LC")":"$(dirname "$LB")");; esac

set -x
"$CXX" -O3 $ARCH -std=c++17 -I"$L/ggml/include" "$HERE/bench_repack.cpp" -o "$OUT" \
  -Wl,--start-group "$LG" "$LC" "$LB" -Wl,--end-group "${RPATH[@]}" "${EXTRA[@]}" -lpthread -ldl -lm
