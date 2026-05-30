#!/usr/bin/env bash
# scripts/build_ncnn_arm_heavy.sh — one-time prebuild of ncnn's ARM-heavy conv
# sources into a static archive that every baseline-ncnn-arm Solution links
# against (PHASE2.md deliverable #1).
#
# Output: arm-bench/build/libncnn_arm_heavy.a  (gitignored)
#
# Inputs (Phase 2 = conv2d only; Phase 3 extends the SOURCES array):
#   - $BASE_ROOT/arm-heavy-optimized/conv/convolution_arm.cpp
#   - $BASE_ROOT/c-partially-optimized/conv/convolution.cpp          (base Convolution)
#
# Re-runs no-op when the .a is newer than every source. Force a rebuild with
# ARMBENCH_REBUILD=1. ISA defaults to `sve`; override with ARMBENCH_ISA=neon|sve|sve2.
# BASE_ROOT defaults to ../ncnn (matches bench/compile.py); override with
# ARMBENCH_BASE_ROOT=<path>.

set -euo pipefail

ARM_BENCH_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE_ROOT="${ARMBENCH_BASE_ROOT:-$ARM_BENCH_ROOT/../ncnn}"
BUILD_DIR="$ARM_BENCH_ROOT/build"
OUT="$BUILD_DIR/libncnn_arm_heavy.a"

SOURCES=(
    "$BASE_ROOT/arm-heavy-optimized/conv/convolution_arm.cpp"
    "$BASE_ROOT/c-partially-optimized/conv/convolution.cpp"
)

for s in "${SOURCES[@]}"; do
    if [[ ! -f "$s" ]]; then
        echo "ERROR: source not found: $s" >&2
        echo "Set ARMBENCH_BASE_ROOT=<ncnn checkout> if your layout differs." >&2
        exit 1
    fi
done

mkdir -p "$BUILD_DIR"

# Skip if .a is newer than all sources.
if [[ -f "$OUT" && "${ARMBENCH_REBUILD:-0}" != "1" ]]; then
    newest=$(stat -c %Y "${SOURCES[@]}" | sort -n | tail -1)
    out_mtime=$(stat -c %Y "$OUT")
    if (( out_mtime >= newest )); then
        echo "$OUT is up-to-date (skip rebuild). Set ARMBENCH_REBUILD=1 to force."
        exit 0
    fi
fi

ISA="${ARMBENCH_ISA:-sve}"
case "$ISA" in
    neon) MARCH="-march=armv8.2-a+fp16+dotprod" ;;
    sve)  MARCH="-march=armv8.2-a+fp16+dotprod+sve" ;;
    sve2) MARCH="-march=armv8.2-a+fp16+dotprod+sve2" ;;
    *) echo "ERROR: unknown ARMBENCH_ISA=$ISA (want neon|sve|sve2)" >&2; exit 1 ;;
esac

CXX="${CXX:-clang++}"

OBJS=()
for src in "${SOURCES[@]}"; do
    obj="$BUILD_DIR/$(basename "${src%.cpp}").o"
    OBJS+=("$obj")
    echo "  CC  $(basename "$src")"
    "$CXX" -c -O2 -fPIC -std=c++14 -fopenmp $MARCH \
        -I "$BASE_ROOT" \
        -I "$BASE_ROOT/framework" \
        -I "$BASE_ROOT/arm-heavy-optimized/conv" \
        -I "$BASE_ROOT/arm-heavy-optimized/common" \
        -I "$BASE_ROOT/c-partially-optimized/conv" \
        -I "$BASE_ROOT/c-partially-optimized/common" \
        -o "$obj" "$src"
done

ar rcs "$OUT" "${OBJS[@]}"
echo "  AR  $OUT"
