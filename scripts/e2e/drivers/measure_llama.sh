#!/usr/bin/env bash
# measure_llama.sh -- waits for the 3 Llama Fable lanes, then the Llama-3.1-8B e2e table with the SAME protocol as
# measure5 (stock / norepack / agent, t 1/4/16, 5 reps, ppl 64 chunks, -fa off sweeps, tg r=10), then the
# per-kernel comparison vs llama.cpp repack on the same box. One c8g.4xlarge, torn down at the end.
set -u
cd ~/arm-bench-e2e
source ~/miniforge3/etc/profile.d/conda.sh; conda activate armbench-run
export WANDB_ENTITY=ArmBench
LABEL=e2e-llama-measure
E=~/e2e; OUT=$E/llama8b/measure; mkdir -p $OUT
RUNS=~/arm-bench-e2e/agent-runs-e2e-llama8b-fable
MAN=$E/llama8b/manifest.json
M=/home/ubuntu/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
NEED=16
say(){ echo "[ml $(date -u +%H:%M:%S)] $*"; }
live_quota(){ local q; q=$(aws service-quotas get-service-quota --service-code ec2 --quota-code L-1216C47A --query "Quota.Value" --output text 2>/dev/null | cut -d. -f1); case "$q" in (""|*[!0-9]*) echo 32;; (*) echo "$q";; esac; }
vcpus_in_use(){ aws ec2 describe-instances --filters "Name=instance-state-name,Values=running,pending" --query "Reservations[].Instances[].InstanceType" --output text 2>/dev/null | tr "\t" "\n" | grep . | awk "/\\.medium\$/{n+=2;next} /\\.large\$/{n+=2;next} /\\.xlarge\$/{n+=4;next} /\\.2xlarge\$/{n+=8;next} /\\.4xlarge\$/{n+=16;next} /\\.8xlarge\$/{n+=32;next} {n+=4} END{print n+0}"; }

say "waiting for the 3 Llama lanes to finish"
while ps -eo args | grep -q "^e2e_llama_lane"; do sleep 120; done
TS=$(date -u +%Y%m%dT%H%MZ)
say "lanes finished"

say "building the manifest from $RUNS"
python scripts/e2e/build_manifest.py --runs $RUNS --isa sve2 --rows-abi --out $MAN > $OUT/manifest_$TS.log 2>&1 || { say "manifest FAILED"; tail -20 $OUT/manifest_$TS.log; exit 1; }
tail -12 $OUT/manifest_$TS.log
n=$(python -c "import json;print(len(json.load(open(\"$MAN\"))[\"kernels\"]))")
say "manifest has $n of 7 kernels"
[ "$n" -lt 1 ] && { say "no kernels -- stopping"; exit 1; }

waited=0
while :; do
  Q=$(live_quota); used=$(vcpus_in_use); used=${used:-0}
  [ $((used + NEED)) -le $Q ] && { say "${used}/${Q} vCPU in use; room"; break; }
  [ $((waited % 900)) -eq 0 ] && say "${used}/${Q} vCPU, need ${NEED} -- waiting"
  sleep 60; waited=$((waited+60)); [ $waited -ge 14400 ] && { say "gave up waiting for quota"; exit 1; }
done
say "provisioning c8g.4xlarge"
attempt=0
until python eval/provision.py --isa sve2 --instance c8g.4xlarge --on-demand --label $LABEL --dataset llama.cpp > $OUT/prov_$TS.log 2>&1; do
  attempt=$((attempt+1)); [ $attempt -ge 6 ] && { say "provision gave up"; exit 1; }
  say "provision attempt $attempt failed -- retry in 3 min"; sleep 180
done
BOX=$(python -c "
import json, pathlib
f = pathlib.Path.home()/\"arm-bench-e2e/eval/eval_config.json\"
print((json.load(open(f)).get(\"instances\",{}).get(\"$LABEL\") or {}).get(\"host\",\"\"))")
[ -z "$BOX" ] && { say "no box ip"; exit 1; }
SSHB="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -o ServerAliveInterval=30 ubuntu@$BOX"
say "box $BOX"
$SSHB "sudo shutdown -c 2>/dev/null; sudo shutdown -h +360" >/dev/null 2>&1 || true
python scripts/e2e/provision_e2e.py --label $LABEL --model llama-3.1-8b --agent-build > $OUT/e2e_prov_$TS.log 2>&1 || { say "provision_e2e FAILED (box kept)"; tail -15 $OUT/e2e_prov_$TS.log; exit 1; }

say "shipping kernels and scripts"
$SSHB "mkdir -p ~/e2e/llama8b ~/arm-bench/scripts/e2e ~/kvr"
rsync -az -e "ssh -o StrictHostKeyChecking=accept-new" $E/llama8b/ ubuntu@$BOX:e2e/llama8b/ --include="*/" --include="*.so" --include="manifest.json" --exclude="*" || { say "kernel sync FAILED (box kept)"; exit 1; }
rsync -az -e "ssh -o StrictHostKeyChecking=accept-new" ~/arm-bench-e2e/scripts/e2e/ ubuntu@$BOX:arm-bench/scripts/e2e/ || { say "script sync FAILED (box kept)"; exit 1; }
rsync -az -e "ssh -o StrictHostKeyChecking=accept-new" ~/arm-bench-e2e/scripts/e2e/kernel_vs_repack/ ubuntu@$BOX:kvr/
$SSHB "test -f ~/models/wiki.test.raw || (cd /tmp && curl -sL -o wt.zip https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip && python3 -c \"import zipfile; zipfile.ZipFile(\\\"/tmp/wt.zip\\\").extractall(\\\"/tmp/wt\\\")\" && cp /tmp/wt/wikitext-2-raw/wiki.test.raw ~/models/wiki.test.raw)"
$SSHB "test -f ~/models/wiki.test.raw" || { say "wikitext MISSING (box kept)"; exit 1; }

say "hook check"
$SSHB "ARMBENCH_OVERRIDES=$MAN ARMBENCH_OVERRIDE_LOG=1 ~/llama.cpp-e2e/build-agent/bin/llama-bench -m $M -p 16 -n 0 -t 4 -r 1" > $OUT/hook_$TS.log 2>&1
h=$(grep -c "hit mul_mat" $OUT/hook_$TS.log); say "hook: $h shapes intercepted (manifest $n)"
[ "$h" -lt "$n" ] && { say "HOOK DID NOT FIRE for all ($h/$n) -- box $BOX kept"; grep -aiE "armbench|error" $OUT/hook_$TS.log | head -5; exit 1; }

say "1/3 core table + perplexity at 64 chunks (3 arms)"
$SSHB "cd ~/arm-bench && python3 scripts/e2e/measure_e2e.py --model $M \
  --build stock=/home/ubuntu/llama.cpp-e2e/build-stock \
  --build norepack=/home/ubuntu/llama.cpp-e2e/build-norepack \
  --build fable=/home/ubuntu/llama.cpp-e2e/build-agent \
  --overrides fable=$MAN \
  --threads 1 4 16 --pp 512 --tg 128 --reps 5 \
  --perplexity /home/ubuntu/models/wiki.test.raw --ppl-chunks 64 --out /home/ubuntu/e2e_llama.json" > $OUT/measure_stdout_$TS.log 2>&1
ok=0; for try in 1 2 3 4 5; do scp -q ubuntu@$BOX:e2e_llama.json $OUT/e2e_llama_$TS.json && ok=1 && break; sleep 10; done
[ "$ok" != "1" ] && { say "COULD NOT COLLECT core table -- box $BOX kept"; tail -20 $OUT/measure_stdout_$TS.log; exit 2; }
say "core results -> $OUT/e2e_llama_$TS.json"
grep -A40 "config  *median" $OUT/measure_stdout_$TS.log | head -30
grep -i "perplexity .*Final" $OUT/measure_stdout_$TS.log | sed "s/.*perplexity /ppl /" | cut -c1-120

bench() { local name="$1"; shift
  for arm in stock agent; do
    if [ "$arm" = stock ]; then B=~/llama.cpp-e2e/build-stock/bin/llama-bench; EV=""; else B=~/llama.cpp-e2e/build-agent/bin/llama-bench; EV="ARMBENCH_OVERRIDES=$MAN"; fi
    say "  $name / $arm"
    $SSHB "$EV $B -m $M -t 16 -o json $*" >> $OUT/${name}_${arm}_$TS.json 2>>$OUT/err_$TS.log || say "  $name/$arm FAILED"
  done
}
say "2/3 sweeps on the fast attention path"
bench pp     -r 3  -p 512,2048,4096,8192 -n 0 -fa off
bench ubatch -r 3  -p 512 -n 0 -ub 1,2,8,64,512 -fa off
bench tg     -r 10 -p 0 -n 128 -fa off

say "3/3 per-kernel comparison vs llama.cpp repack"
$SSHB "cd ~/kvr && ./build.sh ~/llama.cpp-e2e ~/llama.cpp-e2e/build-stock" > $OUT/kvr_build_$TS.log 2>&1 || say "kvr build FAILED"
$SSHB "cd ~/kvr && timeout 3600 ./run_all.sh $MAN /home/ubuntu/kvr_llama.jsonl" > $OUT/kvr_run_$TS.log 2>&1 || say "kvr run non-zero"
scp -q ubuntu@$BOX:kvr_llama.jsonl $OUT/kvr_llama_$TS.jsonl || say "kvr pull FAILED"
tail -8 $OUT/kvr_run_$TS.log

say "table"
python scripts/e2e/results_table.py $OUT/e2e_llama_$TS.json 2>&1 | tee $OUT/table_$TS.txt
python scripts/e2e/wandb_log_e2e.py $OUT/e2e_llama_$TS.json --project arm-bench-kernels-gpt5.6-luna --entity ArmBench \
  --manifest $MAN --group e2e__llama-3.1-8b --name e2e_llama-3.1-8b_c8g4xl_$TS --tag gate-v2 --tag fable-authored --tag ppl64 --tag fa-off \
  --notes "Llama-3.1-8B e2e: Fable gate-v2 kernels vs stock and norepack, same protocol as the Qwen3.5-4B definitive run" 2>&1 | tail -2 || say "wandb logging failed"

say "tearing down"
python eval/provision.py --teardown --label $LABEL > $OUT/teardown_$TS.log 2>&1 || say "TEARDOWN FAILED -- DESTROY $LABEL MANUALLY"
say "measure_llama done -> $OUT"
