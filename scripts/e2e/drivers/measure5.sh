#!/usr/bin/env bash
# measure5.sh — THE definitive end-to-end table for the paper. One box, one kernel set,
# every axis that previously moved the headline settled in the same run.
#
# What this fixes relative to measure4 + the ad-hoc sweeps that followed it:
#   * the hand-patched ("fable_repaired") arm is GONE. Arms are stock / norepack / fable_gatev2.
#   * perplexity at 64 chunks, not 8. The 8-chunk ladder would not order itself (Q4_K_S beat
#     Q4_K_M), which disqualifies it as a reference; 64 chunks orders correctly.
#   * the override carries the M<nth dispatch fix, so micro-batch is not measured on a path
#     where 2 of 16 threads did all the work.
#   * the prompt-length curve is measured with -fa off, the fast attention path on this CPU.
#     `-fa auto` picked the ~2x slower path and made the whole decay curve an artifact.
#   * decode gets -r 10 instead of -r 3, to settle whether tg is 1.025x or 1.074x.
#
# Capacity-gated: a c8g.4xlarge is 16 vCPU against a 32-vCPU quota, so this waits for room
# rather than failing to provision while the S7 lanes are up.
set -u
cd ~/arm-bench-e2e
source ~/miniforge3/etc/profile.d/conda.sh; conda activate armbench-run
export WANDB_ENTITY=ArmBench
LABEL=e2e-definitive
E=~/e2e; TS=$(date -u +%Y%m%dT%H%MZ); OUT=$E/definitive; mkdir -p $OUT $E/gatev2b
ORIG=~/runs-fable-orig
NEW=~/arm-bench-e2e/agent-runs-e2e-qwen35-4b-fable2
NEWDEFS="gemm_ggml_q4_K_n4096_k2560 gemm_ggml_q4_K_n8192_k2560 gemm_ggml_q4_K_n9216_k2560 gemm_ggml_q5_K_n2560_k4096 gemm_ggml_q5_K_n8192_k2560"
OLDDEFS="gemm_ggml_q4_K_n1024_k2560 gemm_ggml_q4_K_n2560_k4096 gemm_ggml_q4_K_n2560_k9216 gemm_ggml_q6_K_n1024_k2560 gemm_ggml_q6_K_n248320_k2560 gemm_ggml_q6_K_n2560_k9216"
MAN=$E/gatev2b/manifest_gatev2b.json
M=/home/ubuntu/models/Qwen3.5-4B-Q4_K_M.gguf
QUOTA=32; NEED=16
say(){ echo "[m5 $(date -u +%H:%M:%S)] $*"; }

vcpus_in_use(){
  aws ec2 describe-instances --filters "Name=instance-state-name,Values=running,pending" \
    --query "Reservations[].Instances[].InstanceType" --output text 2>/dev/null \
    | tr '\t' '\n' | grep . | awk '
      /\.medium$/ {n+=2; next} /\.large$/ {n+=2; next} /\.xlarge$/ {n+=4; next}
      /\.2xlarge$/{n+=8; next} /\.4xlarge$/{n+=16; next} /\.8xlarge$/{n+=32; next}
      {n+=4} END{print n+0}'
}

# ---- everything that can fail cheaply happens BEFORE we ask for a box ----
say "building the gate-v2 manifest from run dirs (6 reused + 5 re-optimized)"
python scripts/e2e/build_manifest.py --runs $ORIG $NEW --isa sve2 --rows-abi --out $MAN \
  || { say "manifest FAILED"; exit 1; }
n=$(python -c "import json;print(len(json.load(open('$MAN'))['kernels']))")
say "manifest has $n kernels"
[ "$n" -ne 11 ] && { say "expected 11 kernels, got $n -- stopping"; exit 1; }

say "VERIFYING the manifest really holds the re-optimized kernels"
python scripts/e2e/verify_kernel_set.py --manifest $MAN --reference $ORIG \
  --expect-count 11 --expect-changed $NEWDEFS --expect-same $OLDDEFS \
  --forbid 'maxabs >> sh' 2>&1 | tee $OUT/verify_$TS.log
[ "${PIPESTATUS[0]}" -ne 0 ] && { say "KERNEL SET VERIFICATION FAILED -- nothing provisioned"; exit 1; }

grep -q "M < nth" scripts/e2e/override/armbench_override.c \
  || { say "override is missing the M<nth dispatch fix -- nothing provisioned"; exit 1; }
say "override carries the dispatch fix"

waited=0
while :; do
  used=$(vcpus_in_use); used=${used:-0}
  [ $((used + NEED)) -le $QUOTA ] && { say "${used}/${QUOTA} vCPU in use; room for a c8g.4xlarge"; break; }
  [ $((waited % 900)) -eq 0 ] && say "${used}/${QUOTA} vCPU in use, need ${NEED} -- waiting (${waited}s)"
  sleep 60; waited=$((waited+60))
  [ $waited -ge 28800 ] && { say "gave up waiting for quota after 8h"; exit 1; }
done

say "provisioning c8g.4xlarge"
python eval/provision.py --isa sve2 --instance c8g.4xlarge --on-demand --label $LABEL --dataset llama.cpp || { say "provision FAILED"; exit 1; }
BOX=$(python -c "
import json, pathlib
for p in ('arm-bench-e2e/eval/eval_config.json','arm-bench/eval/eval_config.json'):
    f = pathlib.Path.home()/p
    if not f.exists(): continue
    h = (json.load(open(f)).get('instances',{}).get('$LABEL') or {}).get('host','')
    if h: print(h); break
")
[ -z "$BOX" ] && { say "no box ip"; exit 1; }
SSHB="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 ubuntu@$BOX"
say "box $BOX"
$SSHB 'sudo shutdown -c 2>/dev/null; sudo shutdown -h +300' || true
python scripts/e2e/provision_e2e.py --label $LABEL --model qwen3.5-4b --agent-build || { say "provision_e2e FAILED (box kept)"; exit 1; }

say "shipping kernels and scripts"
$SSHB 'mkdir -p ~/e2e/gatev2b ~/arm-bench/scripts/e2e' || { say "mkdir on box FAILED -- box $BOX kept"; exit 1; }
rsync -az -e "ssh -o StrictHostKeyChecking=accept-new" --include '*.so' --include 'manifest_gatev2b.json' --exclude '*' $E/gatev2b/ ubuntu@$BOX:e2e/gatev2b/ \
  || { say "kernel sync FAILED -- box $BOX kept"; exit 1; }
rsync -az -e "ssh -o StrictHostKeyChecking=accept-new" ~/arm-bench-e2e/scripts/e2e/ ubuntu@$BOX:arm-bench/scripts/e2e/ \
  || { say "script sync FAILED -- box $BOX kept"; exit 1; }
$SSHB "test -f ~/arm-bench/scripts/e2e/measure_e2e.py" || { say "measure_e2e.py MISSING on box -- box $BOX kept"; exit 1; }
$SSHB 'test -f ~/models/wiki.test.raw || (cd /tmp && curl -sL -o wt.zip https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip && python3 -c "
import zipfile; zipfile.ZipFile(\"/tmp/wt.zip\").extractall(\"/tmp/wt\")" && cp /tmp/wt/wikitext-2-raw/wiki.test.raw ~/models/wiki.test.raw)'
$SSHB 'test -f ~/models/wiki.test.raw' || { say "wikitext MISSING -- box $BOX kept"; exit 1; }

say "hook check"
$SSHB "ARMBENCH_OVERRIDES=$MAN ARMBENCH_OVERRIDE_LOG=1 ~/llama.cpp-e2e/build-agent/bin/llama-bench -m $M -p 16 -n 0 -t 4 -r 1" > $OUT/hook_$TS.log 2>&1
h=$(grep -c 'hit mul_mat' $OUT/hook_$TS.log); say "hook: $h shapes intercepted"
[ "$h" -lt 11 ] && { say "HOOK DID NOT FIRE ($h/11) -- box $BOX kept"; grep -aiE 'armbench|error' $OUT/hook_$TS.log | head -3; exit 1; }
say "dispatch modes:"; grep -o 'dispatch=[a-zA-Z-]*' $OUT/hook_$TS.log | sort | uniq -c | sed 's/^/    /'

say "1/2 core table + perplexity at 64 chunks (3 arms)"
$SSHB "cd ~/arm-bench && python3 scripts/e2e/measure_e2e.py --model $M \
  --build stock=/home/ubuntu/llama.cpp-e2e/build-stock \
  --build norepack=/home/ubuntu/llama.cpp-e2e/build-norepack \
  --build fable_gatev2=/home/ubuntu/llama.cpp-e2e/build-agent \
  --overrides fable_gatev2=$MAN \
  --threads 1 4 16 --pp 512 --tg 128 --reps 5 \
  --perplexity /home/ubuntu/models/wiki.test.raw --ppl-chunks 64 --out /home/ubuntu/e2e_definitive.json" 2>&1 | tee $OUT/measure_stdout_$TS.log

ok=0
for try in 1 2 3 4 5; do
  scp -q ubuntu@$BOX:~/e2e_definitive.json $OUT/e2e_definitive_$TS.json && ok=1 && break
  say "scp attempt $try failed"; sleep 10
done
[ "$ok" != "1" ] && { say "COULD NOT COLLECT core table -- box $BOX kept"; exit 2; }
say "core results -> $OUT/e2e_definitive_$TS.json"

bench() { local name="$1"; shift
  for arm in stock agent; do
    if [ "$arm" = stock ]; then B=~/llama.cpp-e2e/build-stock/bin/llama-bench; EV=""; else B=~/llama.cpp-e2e/build-agent/bin/llama-bench; EV="ARMBENCH_OVERRIDES=$MAN"; fi
    say "  $name / $arm"
    $SSHB "$EV $B -m $M -t 16 -o json $*" >> $OUT/${name}_${arm}_$TS.json 2>>$OUT/err_$TS.log || say "  $name/$arm FAILED"
  done
}
say "2/2 sweeps on the fast attention path"
bench pp     -r 3  -p 512,2048,4096,8192 -n 0 -fa off
bench ubatch -r 3  -p 512 -n 0 -ub 1,2,8,64,512 -fa off
bench tg     -r 10 -p 0 -n 128 -fa off

say "table"
python scripts/e2e/results_table.py $OUT/e2e_definitive_$TS.json 2>&1 | tee $OUT/table_$TS.txt
if [ -f ~/sweep_table2.py ]; then python ~/sweep_table2.py $OUT 2>&1 | tee -a $OUT/table_$TS.txt || say "(sweep table failed; raw json kept)"; else say "(no sweep_table2.py; raw llama-bench json kept in $OUT)"; fi

python scripts/e2e/wandb_log_e2e.py $OUT/e2e_definitive_$TS.json \
  --project arm-bench-kernels-gpt5.6-luna --entity ArmBench \
  --manifest $MAN --group e2e__qwen3.5-4b \
  --name e2e_qwen3.5-4b_definitive_c8g4xl_$TS --tag gate-v2 --tag fable-authored --tag definitive --tag ppl64 --tag fa-off \
  --notes "definitive e2e table: Fable gate-v2 set vs stock and norepack, dispatch fix, -fa off, perplexity at 64 chunks, decode at r=10" \
  2>&1 | tail -3 || say "wandb logging failed (results on disk)"

say "tearing down"
python eval/provision.py --teardown --label $LABEL || say "TEARDOWN FAILED -- DESTROY $LABEL MANUALLY"
say "measure5 done -> $OUT"
