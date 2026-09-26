#!/usr/bin/env bash
# profile_ops.sh — measured (not byte-estimated) per-kernel time shares of a decode/prefill run,
# via perf on the box. Complements scripts/e2e/qwen35_inventory.py (byte shares): the memory-bound
# decode should track bytes, but prefill (pp) is compute-bound and the GDN/attention/norm ops only
# show up here. Output: perf report sorted by symbol, plus a bucketed summary by kernel family.
#   scripts/e2e/profile_ops.sh <build-dir> <model.gguf> <threads> [n_gen=64] [n_prompt=0]
# Needs: sudo apt-get install -y linux-tools-common linux-tools-$(uname -r)   (perf), and the build
# compiled with symbols (default cmake RelWithDebInfo is fine; -DCMAKE_BUILD_TYPE=Release keeps symbols).
set -euo pipefail
B="${1:?build dir}"; M="${2:?model}"; T="${3:?threads}"; NG="${4:-64}"; NP="${5:-0}"
OUT="${OUT:-profile_$(date +%Y%m%d_%H%M%S)}"; mkdir -p "$OUT"
if [ "$NP" -gt 0 ]; then ARGS=(-p "$NP" -n 0 -fa 0); KIND=pp; else ARGS=(-p 0 -n "$NG" -fa 0); KIND=tg; fi
"${PERF:-perf}" record -F 999 -g -o "$OUT/perf.data" -- "$B/bin/llama-bench" -m "$M" -t "$T" "${ARGS[@]}" -r 3 -o json > "$OUT/llama-bench.json" 2> "$OUT/llama-bench.err"
"${PERF:-perf}" report -i "$OUT/perf.data" --no-children --sort symbol --stdio 2>/dev/null | grep -v '^#' | grep -v '^$' > "$OUT/symbols.txt"
python3 - "$OUT/symbols.txt" "$KIND" <<'PY'
import re, sys, collections
buckets = collections.OrderedDict([
  ("mul_mat q4_K",  r"q4_K"), ("mul_mat q5_K", r"q5_K"), ("mul_mat q6_K", r"q6_K"), ("mul_mat q8_0", r"q8_0"),
  ("mul_mat other/repack", r"mul_mat|gemm|gemv|repack|kleidi|kai_"), ("quantize act (q8_K)", r"quantize_row_q8"),
  ("gated_delta_net", r"gated_delta|delta_net"), ("ssm_conv", r"ssm_conv"), ("attention", r"flash_attn|soft_max"),
  ("rms_norm/l2_norm", r"rms_norm|l2_norm|norm"), ("rope", r"rope"), ("glu/silu/act", r"glu|silu|swiglu|gelu"),
  ("get_rows/cpy/add/mul", r"get_rows|cpy|dup|add|_mul\b|binary"), ("scheduler/threads/other ggml", r"ggml|graph|thread"),
])
tot = collections.Counter(); other = 0.0; n = 0
for line in open(sys.argv[1]):
    m = re.match(r"\s*([\d.]+)%\s+(.*)", line)
    if not m: continue
    pct, sym = float(m.group(1)), m.group(2); n += 1
    for name, rx in buckets.items():
        if re.search(rx, sym, re.I): tot[name] += pct; break
    else: other += pct
print(f"perf symbol buckets ({sys.argv[2]}, {n} symbols)")
for name, _ in buckets.items():
    if tot[name] > 0.05: print(f"  {name:28} {tot[name]:6.1f}%")
print(f"  {'other (libc, kernel, ...)':28} {other:6.1f}%")
PY
echo "raw: $OUT/symbols.txt   bench: $OUT/llama-bench.json"
