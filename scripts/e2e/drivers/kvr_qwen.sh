#!/usr/bin/env bash
# kvr_qwen.sh -- per-kernel timing: Fable (gatev2b) vs llama.cpp repacked vs plain ggml, 11 Qwen3.5-4B shapes,
# single-threaded, one c8g.4xlarge, torn down at the end.
set -u
cd ~/arm-bench-e2e
source ~/miniforge3/etc/profile.d/conda.sh; conda activate armbench-run
LABEL=e2e-kvr; TS=$(date -u +%Y%m%dT%H%MZ); OUT=~/e2e/kvr; mkdir -p $OUT/logs
say(){ echo "[kvr $(date -u +%H:%M:%S)] $*"; }
teardown(){ say "tearing down"; python eval/provision.py --teardown --label $LABEL > $OUT/logs/teardown_$TS.log 2>&1 || say "TEARDOWN FAILED -- DESTROY $LABEL MANUALLY"; }
say "provisioning c8g.4xlarge"
attempt=0
until python eval/provision.py --isa sve2 --instance c8g.4xlarge --on-demand --label $LABEL --dataset llama.cpp > $OUT/logs/prov_$TS.log 2>&1; do
  attempt=$((attempt+1)); [ $attempt -ge 6 ] && { say "provision gave up"; teardown; exit 1; }
  say "provision attempt $attempt failed -- retry in 3 min"; sleep 180
done
BOX=$(python -c "
import json, pathlib
f = pathlib.Path.home()/\"arm-bench-e2e/eval/eval_config.json\"
print((json.load(open(f)).get(\"instances\",{}).get(\"$LABEL\") or {}).get(\"host\",\"\"))")
[ -z "$BOX" ] && { say "no box ip"; teardown; exit 1; }
SSHB="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o ServerAliveInterval=30 ubuntu@$BOX"
say "box $BOX"
$SSHB "sudo shutdown -c 2>/dev/null; sudo shutdown -h +150" >/dev/null 2>&1 || true
python scripts/e2e/provision_e2e.py --label $LABEL > $OUT/logs/e2e_prov_$TS.log 2>&1 || { say "provision_e2e FAILED"; tail -15 $OUT/logs/e2e_prov_$TS.log; teardown; exit 1; }
say "llama.cpp stock built; shipping harness + kernels"
$SSHB "mkdir -p ~/e2e/gatev2b ~/kvr" || { teardown; exit 1; }
rsync -az -e "ssh -o StrictHostKeyChecking=accept-new" --include="*.so" --include="manifest_gatev2b.json" --exclude="*" ~/e2e/gatev2b/ ubuntu@$BOX:e2e/gatev2b/ || { say "kernel sync FAILED"; teardown; exit 1; }
rsync -az -e "ssh -o StrictHostKeyChecking=accept-new" scripts/e2e/kernel_vs_repack/ ubuntu@$BOX:kvr/ || { say "harness sync FAILED"; teardown; exit 1; }
say "building harness"
$SSHB "cd ~/kvr && ./build.sh ~/llama.cpp-e2e ~/llama.cpp-e2e/build-stock" > $OUT/logs/build_$TS.log 2>&1 || { say "harness build FAILED"; tail -20 $OUT/logs/build_$TS.log; teardown; exit 1; }
say "running (warm cache)"
$SSHB "cd ~/kvr && timeout 3600 ./run_all.sh /home/ubuntu/e2e/gatev2b/manifest_gatev2b.json /home/ubuntu/kvr_qwen_warm.jsonl" > $OUT/logs/run_warm_$TS.log 2>&1 || say "warm run non-zero (see log)"
scp -o StrictHostKeyChecking=accept-new ubuntu@$BOX:kvr_qwen_warm.jsonl $OUT/kvr_qwen_warm_$TS.jsonl || say "pull warm FAILED"
tail -30 $OUT/logs/run_warm_$TS.log
teardown
say "kvr done -> $OUT"
