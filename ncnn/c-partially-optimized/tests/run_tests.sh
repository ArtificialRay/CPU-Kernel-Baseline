#!/usr/bin/env bash
# run_tests.sh — build and run all ncnn kernel unit tests
# Usage:  bash run_tests.sh [build_dir]

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${1:-${SCRIPT_DIR}/build}"

echo "=== Building tests in: ${BUILD_DIR} ==="
mkdir -p "${BUILD_DIR}"
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release -Wno-dev > /dev/null
cmake --build "${BUILD_DIR}" --parallel "$(nproc 2>/dev/null || echo 4)"

echo ""
echo "=== Running tests ==="
TESTS=(
    test_activation
    test_conv
    test_gemm
    test_norm
    test_attention
    test_recurrent
    test_reduction
    test_tensor
    test_quant
    test_other
)

PASS=0
FAIL=0
for t in "${TESTS[@]}"; do
    exe="${BUILD_DIR}/${t}"
    if [ ! -f "${exe}" ]; then
        echo "  MISSING  ${t}"
        ((FAIL++))
        continue
    fi
    if "${exe}"; then
        ((PASS++))
    else
        ((FAIL++))
    fi
    echo ""
done

echo "=============================="
echo "Suites passed: ${PASS} / $((PASS + FAIL))"
echo "=============================="
[ "${FAIL}" -eq 0 ]
