#!/usr/bin/env bash
# profile_run.sh — measured per-op time shares for stock vs agent, at two prompt lengths and
# at decode. This is the missing evidence behind the decay curve: we show the speedup falls
# from 1.377x @512 to 1.279x @8192 and assert it is because the ops the hook does NOT replace
# (attention, GDN recurrence, norms) grow as a share of runtime. Nothing has measured that.
#
# profile_ops.sh perf-records one llama-bench run and buckets symbols by op family.
set -u
cd ~/arm-bench-e2e
source ~/miniforge3/etc/profile.d/conda.sh; conda activate armbench-run
LABEL=e2e-profile2; TS=$(date -u +%Y%m%dT%H%MZ); OUT=~/e2e/profile; mkdir -p $OUT
E=~/e2e; MAN=$E/gatev2b/manifest_gatev2b.json
M=/home/ubuntu/models/Qwen3.5-4B-Q4_K_M.gguf
NEED=16
say(){ echo "[prof $(date -u +%H:%M:%S)] $*"; }

live_quota(){ local q; q=$(aws service-quotas get-service-quota --service-code ec2 \
  --quota-code L-1216C47A --query "Quota.Value" --output text 2>/dev/null | cut -d. -f1)
  case "$q" in (''|*[!0-9]*) echo 32;; (*) echo "$q";; esac; }
vcpus_in_use(){ aws ec2 describe-instances --filters "Name=instance-state-name,Values=running,pending" \
  --query "Reservations[].Instances[].InstanceType" --output text 2>/dev/null \
  | tr '\t' '\n' | grep . | awk '/\.medium$/{n+=2;next} /\.large$/{n+=2;next} /\.xlarge$/{n+=4;next}
      /\.2xlarge$/{n+=8;next} /\.4xlarge$/{n+=16;next} /\.8xlarge$/{n+=32;next} {n+=4} END{print n+0}'; }

[ -f "$MAN" ] || { say "no manifest at $MAN"; exit 1; }
waited=0
while :; do
  Q=$(live_quota); used=$(vcpus_in_use); used=${used:-0}
  [ $((used + NEED)) -le $Q ] && { say "${used}/${Q} vCPU in use; room"; break; }
  [ $((waited % 900)) -eq 0 ] && say "${used}/${Q} vCPU in use, need ${NEED} -- waiting"
  sleep 60; waited=$((waited+60)); [ $waited -ge 21600 ] && { say "gave up"; exit 1; }
done

say "provisioning c8g.4xlarge"
attempt=0
until python eval/provision.py --isa sve2 --instance c8g.4xlarge --on-demand --label $LABEL --dataset llama.cpp > $OUT/prov_$TS.log 2>&1; do
  attempt=$((attempt+1)); [ $attempt -ge 40 ] && { say "provision gave up"; exit 1; }
  say "provision attempt $attempt failed -- retry in 5 min"; sleep 300
done
BOX=$(python -c "
import json, pathlib
f = pathlib.Path.home()/'arm-bench-e2e/eval/eval_config.json'
print((json.load(open(f)).get('instances',{}).get('$LABEL') or {}).get('host',''))
")
[ -z "$BOX" ] && { say "no box ip"; exit 1; }
SSHB="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 ubuntu@$BOX"
say "box $BOX"
$SSHB 'sudo shutdown -c 2>/dev/null; sudo shutdown -h +180' || true
python scripts/e2e/provision_e2e.py --label $LABEL --model qwen3.5-4b --agent-build || { say "provision_e2e FAILED (box kept)"; exit 1; }

say "shipping kernels and scripts"
$SSHB 'mkdir -p ~/e2e/gatev2b ~/arm-bench/scripts/e2e' || { say "mkdir FAILED -- box kept"; exit 1; }
rsync -az -e "ssh -o StrictHostKeyChecking=accept-new" --include '*.so' --include 'manifest_gatev2b.json' --exclude '*' $E/gatev2b/ ubuntu@$BOX:e2e/gatev2b/ || { say "kernel sync FAILED -- box kept"; exit 1; }
rsync -az -e "ssh -o StrictHostKeyChecking=accept-new" ~/arm-bench-e2e/scripts/e2e/ ubuntu@$BOX:arm-bench/scripts/e2e/ || { say "script sync FAILED -- box kept"; exit 1; }

say "installing perf"
$SSHB 'sudo apt-get update -qq >/dev/null 2>&1; sudo apt-get install -y -qq linux-tools-common "linux-tools-$(uname -r)" >/dev/null 2>&1; which perf || ls /usr/lib/linux-tools/*/perf 2>/dev/null | head -1' > $OUT/perf_install_$TS.log 2>&1
PERFBIN=$($SSHB "sudo apt-get install -y -qq linux-tools-aws >/dev/null 2>&1; ls /usr/lib/linux-tools/*/perf /usr/lib/linux-tools-*/perf 2>/dev/null | head -1"); echo "perfbin=$PERFBIN" >> $OUT/perf_install_$TS.log; [ -n "$PERFBIN" ] && $SSHB "sudo sysctl -q kernel.perf_event_paranoid=-1 kernel.kptr_restrict=0; $PERFBIN --version" >> $OUT/perf_install_$TS.log 2>&1 || { say "perf UNAVAILABLE -- tearing down"; python eval/provision.py --teardown --label $LABEL; exit 1; }
say "perf ok"

prof(){ # prof <arm> <n_prompt> <n_gen>
  local arm="$1" np="$2" ng="$3" tag="$1_p${2}_n${3}"
  local B EV
  if [ "$arm" = stock ]; then B=/home/ubuntu/llama.cpp-e2e/build-stock; EV=""
  else B=/home/ubuntu/llama.cpp-e2e/build-agent; EV="ARMBENCH_OVERRIDES=$MAN"; fi
  say "  profiling $tag"
  $SSHB "cd ~/arm-bench && $EV PERF=$PERFBIN OUT=/home/ubuntu/prof_$tag bash scripts/e2e/profile_ops2.sh $B $M 16 $ng $np" \
    > $OUT/${tag}_$TS.txt 2>&1 || say "    $tag FAILED"
}
for np in 512 8192; do prof stock $np 0; prof agent $np 0; done
prof stock 0 128
prof agent 0 128

say "collecting"
for f in $OUT/*_$TS.txt; do echo "===== $(basename $f)"; sed -n '/perf symbol buckets/,$p' "$f"; done > $OUT/summary_$TS.txt
cat $OUT/summary_$TS.txt

say "tearing down"
python eval/provision.py --teardown --label $LABEL || say "TEARDOWN FAILED -- DESTROY $LABEL MANUALLY"
say "profile done -> $OUT"
