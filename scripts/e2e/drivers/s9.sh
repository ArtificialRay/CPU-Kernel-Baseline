#!/usr/bin/env bash
# s9.sh — measurement-only variance (workbook S9, P1, no LLM).
# Times FIXED code: autovec as candidate against baseline-sve2, repeated N times on one box.
# Nothing about the agent varies, so the spread across repeats is the benchmark's own noise.
# Subtracting this from S7's 15.9% median per-kernel spread separates "the agent behaved
# differently" from "the timer is noisy" -- which decides how hard we can push the
# "ISA differences are inside the noise" claim.
set -u
cd ~/arm-bench
source ~/miniforge3/etc/profile.d/conda.sh; conda activate armbench-run
LABEL=s9-replay; TS=$(date -u +%Y%m%dT%H%MZ); OUT=~/e2e/s9; mkdir -p $OUT
REPS="${1:-3}"
DEFS="loop_001 loop_002 loop_003 loop_010 loop_024 loop_027 loop_035 loop_037 loop_105 loop_108 loop_113 loop_126 loop_127"
say(){ echo "[s9 $(date -u +%H:%M:%S)] $*"; }

say "provisioning c8g.xlarge"
python eval/provision.py --isa sve --instance c8g.xlarge --on-demand --label $LABEL --dataset simd-loop || { say "provision FAILED"; exit 1; }
BOX=$(python -c "
import json, pathlib
for p in ('arm-bench/eval/eval_config.json','arm-bench-e2e/eval/eval_config.json'):
    f = pathlib.Path.home()/p
    if not f.exists(): continue
    h = (json.load(open(f)).get('instances',{}).get('$LABEL') or {}).get('host','')
    if h: print(h); break
")
[ -z "$BOX" ] && { say "no box ip"; exit 1; }
SSHB="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 ubuntu@$BOX"
say "box $BOX"
$SSHB 'sudo shutdown -c 2>/dev/null; sudo shutdown -h +120' || true
# rsync will not create a nested destination path on a fresh box, and swallowing that
# failure is what made the first attempt collect perfect baselines and then time nothing.
$SSHB "mkdir -p ~/arm-bench/scripts/e2e" || { say "cannot mkdir on box"; exit 1; }
rsync -az -e "ssh -o StrictHostKeyChecking=accept-new" ~/arm-bench/scripts/e2e/ ubuntu@$BOX:arm-bench/scripts/e2e/ \
  || { say "script sync FAILED -- box $BOX kept"; exit 1; }
$SSHB "test -f ~/arm-bench/scripts/e2e/speed_compare.py" \
  || { say "speed_compare.py MISSING on box -- box $BOX kept"; exit 1; }

say "collecting baselines once (fixed for every repeat)"
for d in $DEFS; do
  $SSHB "cd ~/arm-bench && python3 -m bench.cli --log-level WARNING collect-baselines --baseline-author baseline-sve2 --definition $d" >> $OUT/baselines_$TS.log 2>&1 || say "  baseline $d FAILED"
done
say "baselines done"

for i in $(seq 1 $REPS); do
  say "repeat $i/$REPS — timing autovec vs baseline-sve2 (identical code each time)"
  $SSHB "cd ~/arm-bench && python3 scripts/e2e/speed_compare.py --authors autovec --definitions $DEFS --baseline-author baseline-sve2 --out /home/ubuntu/s9_rep$i.json" > $OUT/rep${i}_$TS.log 2>&1 \
    || say "  repeat $i FAILED"
  scp -q ubuntu@$BOX:~/s9_rep$i.json $OUT/s9_rep${i}_$TS.json 2>/dev/null || say "  collect rep $i failed"
done

say "tearing down"
python eval/provision.py --teardown --label $LABEL || say "TEARDOWN FAILED -- DESTROY $LABEL MANUALLY"
say "s9 done -> $OUT"
