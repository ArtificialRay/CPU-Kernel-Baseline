#!/usr/bin/env bash
# llama_dump_fix.sh -- redo the Llama-3.1-8B activation capture without llama.cpp warmup (warmup pushes
# 1-2 tokens and was captured instead of the real chunk), then workloads, calibration, pull, teardown.
set -u
cd ~/arm-bench-e2e
source ~/miniforge3/etc/profile.d/conda.sh; conda activate armbench-run
LABEL=e2e-llama-dump; TS=$(date -u +%Y%m%dT%H%MZ); OUT=~/e2e/llama8b; BOX=34.217.49.48
MF=Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
DEFS="gemm_ggml_q4_K_n14336_k4096 gemm_ggml_q6_K_n4096_k14336 gemm_ggml_q4_K_n4096_k4096 gemm_ggml_q4_K_n4096_k14336 gemm_ggml_q6_K_n128256_k4096 gemm_ggml_q4_K_n1024_k4096 gemm_ggml_q6_K_n1024_k4096"
SSHB="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o ServerAliveInterval=30 ubuntu@$BOX"
say(){ echo "[fix $(date -u +%H:%M:%S)] $*"; }
teardown(){ say "tearing down"; python eval/provision.py --teardown --label $LABEL > $OUT/logs/teardown_$TS.log 2>&1 || say "TEARDOWN FAILED -- DESTROY $LABEL MANUALLY"; }

say "resetting the 7 workload files on the box to the generated (random) versions"
for d in $DEFS; do
  rsync -az -e "ssh -o StrictHostKeyChecking=accept-new" bench-trace/workloads/gemm/$d.jsonl ubuntu@$BOX:arm-bench/bench-trace/workloads/gemm/ || { say "reset $d FAILED"; teardown; exit 1; }
  $SSHB "rm -rf ~/arm-bench/bench-trace/tensors/gemm/$d"
done
say "capturing activations (no warmup)"
$SSHB "rm -rf ~/dump; mkdir -p ~/dump; cd ~ && ARMBENCH_OVERRIDES=/home/ubuntu/ref/manifest.json ARMBENCH_OVERRIDE_THREADS=1 ARMBENCH_OVERRIDE_LOG=1 ARMBENCH_DUMP_DIR=/home/ubuntu/dump ARMBENCH_DUMP_CALLS=4 ARMBENCH_DUMP_STRIDE=8 timeout 7200 ~/llama.cpp-e2e/build-agent/bin/llama-perplexity -m ~/models/$MF -f ~/models/wiki.test.raw --chunks 1 -c 128 -b 128 -ub 128 -t 16 --no-warmup" > $OUT/logs/dumprun_$TS.log 2>&1
say "capture exit $? ; $(grep -o "Final estimate.*" $OUT/logs/dumprun_$TS.log | head -1)"
$SSHB "cat ~/dump/*.meta" | python3 -c "
import sys,json,collections
c=collections.defaultdict(list)
for l in sys.stdin: d=json.loads(l); c[(d[\"type\"],d[\"K\"],d[\"N\"])].append(d[\"M\"])
for k,v in sorted(c.items()): print(\"  capture\",k,\"M per call:\",v)
bad=[k for k,v in c.items() if min(v)<32]
print(\"  CAPTURE_OK\" if len(c)==7 and not bad else f\"  CAPTURE_BAD shapes={len(c)} small={bad}\")
" | tee $OUT/logs/capture_$TS.txt
grep -q CAPTURE_OK $OUT/logs/capture_$TS.txt || { say "capture not usable -- box KEPT for inspection ($BOX)"; exit 1; }

say "dump -> workloads"
$SSHB "cd ~/arm-bench && python3 scripts/e2e/dump_to_workloads.py --dump /home/ubuntu/dump --root /home/ubuntu/arm-bench/bench-trace --source \"Llama-3.1-8B-Instruct-Q4_K_M / wikitext-2 test chunk 0\"" > $OUT/logs/d2w_$TS.log 2>&1 || { say "dump_to_workloads FAILED -- box KEPT"; tail -20 $OUT/logs/d2w_$TS.log; exit 1; }
grep -E "workloads ->|skipping|rewritten" $OUT/logs/d2w_$TS.log | grep -v "had no dump" | tail -12
say "calibrating SQNR floors"
$SSHB "cd ~/arm-bench && timeout 5400 python3 scripts/e2e/calibrate_sqnr_floor.py --definitions $DEFS" > $OUT/logs/calib_$TS.log 2>&1 || say "calibrate returned non-zero (see log)"
tail -10 $OUT/logs/calib_$TS.log

say "pulling workloads + tensors back"
mkdir -p $OUT/backup_random_$TS; for d in $DEFS; do cp bench-trace/workloads/gemm/$d.jsonl $OUT/backup_random_$TS/; done
for d in $DEFS; do
  rsync -az -e "ssh -o StrictHostKeyChecking=accept-new" ubuntu@$BOX:arm-bench/bench-trace/workloads/gemm/$d.jsonl bench-trace/workloads/gemm/ || say "pull $d workloads FAILED"
  mkdir -p bench-trace/tensors/gemm/$d
  rsync -az -e "ssh -o StrictHostKeyChecking=accept-new" ubuntu@$BOX:arm-bench/bench-trace/tensors/gemm/$d/ bench-trace/tensors/gemm/$d/ || say "pull $d tensors FAILED"
done
rsync -az -e "ssh -o StrictHostKeyChecking=accept-new" ubuntu@$BOX:dump/ $OUT/dump/ >/dev/null 2>&1
python - <<PY
import json
for d in "$DEFS".split():
    rows=[json.loads(l) for l in open(f"bench-trace/workloads/gemm/{d}.jsonl")]
    real=sum(1 for r in rows if r["inputs"]["A"].get("type")=="tensor")
    cal=sum(1 for r in rows if "baseline_sqnr_db" in json.dumps(r))
    print(f"  {d:34} workloads={len(rows)} real_act={real} calibrated={cal}")
PY
teardown
say "llama prep done"
