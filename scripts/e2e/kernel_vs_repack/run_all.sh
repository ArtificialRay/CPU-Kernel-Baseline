#!/usr/bin/env bash
# run_all.sh <manifest.json> [out.jsonl] -- run bench_repack for every mul_mat kernel in an override manifest
# (manifest_gatev2b.json format: {"kernels":[{"op","type","K","N","so","symbol","abi",...}]}).
# Env: BENCH (binary, default ./bench_repack next to this script), MS (default "1,2,4,8,16,32,128,512"),
#      BIG_N (default 100000) / BIG_MS (default "1,2,4,8"): M list for lm_head-sized shapes (N > BIG_N),
#      BENCH_ARGS (extra args, e.g. "--reps 30 --max-sec 6 --flush-mb 256"), PIN_CPU (default 2; "" = no taskset).
set -euo pipefail
MAN="${1:?manifest.json}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${2:-$HERE/results_$(date +%Y%m%d_%H%M%S).jsonl}"
BENCH="${BENCH:-$HERE/bench_repack}"
MS="${MS:-1,2,4,8,16,32,128,512}"
BIG_N="${BIG_N:-100000}"; BIG_MS="${BIG_MS:-1,2,4,8}"
PIN_CPU="${PIN_CPU-2}"
[ -x "$BENCH" ] || { echo "no bench binary at $BENCH (run build.sh first)" >&2; exit 1; }
unset ARMBENCH_OVERRIDES   # the override hook must not claim the plain-buffer baseline
PIN=(); [ -n "$PIN_CPU" ] && command -v taskset >/dev/null && PIN=(taskset -c "$PIN_CPU")

python3 - "$MAN" <<'EOF' > "$OUT.todo"
import json, sys
for k in json.load(open(sys.argv[1]))["kernels"]:
    if k.get("op", "mul_mat") != "mul_mat":
        continue
    name = k.get("definition") or f"gemm_ggml_{k['type']}_n{k['N']}_k{k['K']}"
    print(k["type"], k["N"], k["K"], k["so"], k.get("symbol", "armbench_llamacpp_gemm"), k.get("abi", "llamacpp"), name)
EOF

echo "[run_all] $(wc -l < "$OUT.todo") shapes -> $OUT" >&2
while read -r TYPE N K SO SYM ABI NAME; do
  M_LIST="$MS"; [ "$N" -gt "$BIG_N" ] && M_LIST="$BIG_MS"
  echo "[run_all] $NAME  M=$M_LIST" >&2
  # shellcheck disable=SC2086
  "${PIN[@]}" "$BENCH" --type "$TYPE" --N "$N" --K "$K" --so "$SO" --symbol "$SYM" --abi "$ABI" \
      --shape "$NAME" --M "$M_LIST" ${BENCH_ARGS:-} >> "$OUT" \
    || echo "{\"shape\":\"$NAME\",\"error\":\"bench exited $?\"}" >> "$OUT"
done < "$OUT.todo"
rm -f "$OUT.todo"

python3 - "$OUT" <<'EOF' >&2
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
f = lambda v, p=1: "-" if v is None else f"{v:.{p}f}"
print(f"{'shape':34} {'M':>4} {'rep':>3} {'repack_us':>10} {'norep_us':>10} {'agent_us':>10} {'a/rep':>6} {'a/norep':>7} {'sqnr':>6}")
for r in rows:
    if "error" in r: print(r["shape"], "ERROR", r["error"]); continue
    print(f"{r['shape']:34} {r['M']:>4} {'y' if r['repack_used'] else 'N':>3} {f(r['t_repack_min_us']):>10} {f(r['t_norepack_min_us']):>10} "
          f"{f(r['t_agent_min_us']):>10} {f(r['agent_vs_repack'],2):>6} {f(r['agent_vs_norepack'],2):>7} {f(r['sqnr_db']):>6}")
EOF
