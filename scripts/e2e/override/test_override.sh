#!/usr/bin/env bash
# test_override.sh [llama.cpp dir] [build dir]
# Builds the test kernel + harness against a llama.cpp checkout that has the override
# patch applied and ggml/ggml-base/ggml-cpu built as static libs, then runs the graph
# once without and once with ARMBENCH_OVERRIDES and compares. Works on macOS (.dylib)
# and Linux (.so).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA="${1:-${LLAMA_CPP_DIR:?usage: test_override.sh <llama.cpp dir> [build dir]}}"
LLAMA="$(cd "$LLAMA" && pwd)"
BUILD="${2:-$LLAMA/build}"
OUT="${TEST_OUT:-$BUILD/armbench_override_test}"
mkdir -p "$OUT"

case "$(uname -s)" in
    Darwin) SOEXT=dylib; EXTRA_LD="";;
    *)      SOEXT=so;    EXTRA_LD="-rdynamic -ldl -lm";;   # -rdynamic: test kernel dlsym()s dequantize_row_q4_K from the host
esac
CC="${CC:-cc}"; CXX="${CXX:-c++}"
LIBS="$BUILD/ggml/src/libggml.a $BUILD/ggml/src/libggml-cpu.a $BUILD/ggml/src/libggml-base.a"
for l in $LIBS; do [ -f "$l" ] || { echo "missing $l (build ggml first)" >&2; exit 1; }; done
grep -q armbench_override_mul_mat "$LLAMA/ggml/src/ggml-cpu/ggml-cpu.c" || { echo "patch not applied in $LLAMA (run apply.sh)" >&2; exit 1; }

echo "== building test kernel (lib test_kernel.$SOEXT)"
$CC -O2 -fPIC -shared -o "$OUT/libtest_kernel.$SOEXT" "$here/test_kernel.c"
echo "== building test harness"
$CXX -std=c++17 -O2 -o "$OUT/test_override" "$here/test_override.cpp" \
    -I"$LLAMA/ggml/include" -I"$here" $LIBS -lpthread $EXTRA_LD

cat > "$OUT/manifest.json" <<JSON
{"kernels":[
  {"op":"mul_mat","type":"q4_K","K":512,"N":64,"so":"$OUT/libtest_kernel.$SOEXT","symbol":"armbench_llamacpp_gemm","abi":"llamacpp"},
  {"op":"mul_mat","type":"q4_K","K":256,"N":32,"so":"$OUT/libtest_kernel.$SOEXT","symbol":"armbench_entry_gemm","abi":"entry","threads":1}
]}
JSON

echo "== run 1: stock path (no ARMBENCH_OVERRIDES)"
env -u ARMBENCH_OVERRIDES "$OUT/test_override" ref "$OUT/ref.bin"

echo "== run 2: override path"
ARMBENCH_OVERRIDES="$OUT/manifest.json" ARMBENCH_OVERRIDE_LOG=1 \
    "$OUT/test_override" check "$OUT/ref.bin" "$OUT/libtest_kernel.$SOEXT" 2> "$OUT/run2.stderr"
echo "-- stderr (ARMBENCH_OVERRIDE_LOG=1):"; sed 's/^/   /' "$OUT/run2.stderr"
n=$(grep -c "hit mul_mat" "$OUT/run2.stderr" || true)
[ "$n" = 2 ] || { echo "expected 2 'hit mul_mat' log lines, got $n" >&2; exit 1; }
echo "== PASS: both ABIs hit, outputs match (artifacts in $OUT)"
