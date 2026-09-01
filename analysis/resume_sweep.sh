#!/usr/bin/env bash
# Idempotent resume for the ncnn+llama.cpp sonnet/sve2 sweep.
# Safe to run ANYTIME (after a crash, sleep, tunnel death, or partial run):
# it recomputes which defs are still incomplete (no 'submit' turn in their
# trajectory.jsonl — the durable completion marker) and re-runs ONLY those,
# looping until everything's done or MAX_ROUNDS. If a dataset makes zero
# progress in a round (box/MCP server wedged), it tears that box down so the
# next round re-provisions a fresh one. Tears down both boxes at the very end.
set -uo pipefail
REPO=/Users/allen/CMU/cpu-kernel-baseline
PY=$(conda run -n armbench-run which python)
AUTHOR=claude-code-sonnet-sve2-40i          # fresh author -> clean results dir + boxes
RESULTS="$REPO/agent-runs-$AUTHOR"
export WANDB=1 WANDB_ENTITY=ArmBench WANDB_PROJECT=arm-bench-kernels WANDB_PYTHON="$PY"
export WANDB_GROUP=sonnet-sve2-40iter
cd "$REPO"
DATASETS="ncnn llama.cpp"
MIN_ITERS=40
MAX_ROUNDS=8

incomplete () {  # $1=dataset -> space-separated incomplete def names
  "$PY" - "$1" "$RESULTS" <<'PYEOF'
import json,sys,glob
from pathlib import Path
ds, results = sys.argv[1], Path(sys.argv[2])
names=[]
for p in glob.glob('bench-trace/definitions/**/*.json',recursive=True):
    d=json.load(open(p)); t=d.get('tags',[])
    x=next((s.split(':',1)[1] for s in t if s.startswith('baseline-solution:')),None)
    if x is None and 'simd-loop' in t: x='simd-loop'
    if x==ds: names.append(d['name'])
def done(n):
    tp=results/n/'trajectory.jsonl'
    if not tp.exists(): return False
    try: return any(json.loads(l).get('tool')=='submit' for l in open(tp) if l.strip())
    except Exception: return False
import re, math
def cost(n):
    # crude workload-size proxy: product of the integers in the def name.
    # pooling/1x1 convs sort first, kh7_kw7 mid, big gemms last — so the sweep
    # yields fast sanity-check data early and saves the slow evals for the tail.
    return math.prod(max(int(x), 1) for x in re.findall(r'\d+', n)) or 1
print(' '.join(sorted((n for n in names if not done(n)), key=lambda n:(cost(n), n))))
PYEOF
}

# kill any lingering driver/tunnel from a broken prior run so they don't fight
pkill -f "sweep_ncnn_llama.sh" 2>/dev/null || true
pkill -f "bench_fleet.py"      2>/dev/null || true
sleep 2

declare -A PREV
for round in $(seq 1 "$MAX_ROUNDS"); do
  ANY=0
  for DS in $DATASETS; do
    MISS=$(incomplete "$DS"); N=$(echo $MISS | wc -w | tr -d ' ')
    if [ "$N" -eq 0 ]; then
      echo "@@@ [$(date +%H:%M:%S)] ROUND $round $DS: all complete"
      continue
    fi
    ANY=1
    LB="$DS-$AUTHOR"
    if [ "${PREV[$DS]:-}" = "$N" ]; then
      echo "@@@ [$(date +%H:%M:%S)] ROUND $round $DS STALLED at $N — tearing down $LB to force a fresh box"
      "$PY" eval/provision.py --teardown --label "$LB" >/dev/null 2>&1 || true
    fi
    PREV[$DS]="$N"
    echo "@@@ [$(date +%H:%M:%S)] ROUND $round $DS incomplete=$N: $MISS"
    "$PY" -u test_scripts/bench_fleet.py --harness claude-code --dataset "$DS" --isa sve2 \
      --model sonnet --author "$AUTHOR" --min-iterations "$MIN_ITERS" --definitions "$MISS" || true
  done
  [ "$ANY" -eq 0 ] && { echo "@@@ ALL COMPLETE at round $round"; break; }
done

for DS in $DATASETS; do
  LEFT=$(incomplete "$DS"); echo "@@@ FINAL $DS still-incomplete: ${LEFT:-none}"
done
for DS in $DATASETS; do
  LB="$DS-$AUTHOR"
  "$PY" eval/provision.py --teardown --label "$LB" >/dev/null 2>&1 \
    && echo "@@@ teardown $LB OK" || echo "@@@ teardown $LB FAILED (destroy manually)"
done
echo "@@@ RESUME_DONE $(date +%H:%M:%S)"
