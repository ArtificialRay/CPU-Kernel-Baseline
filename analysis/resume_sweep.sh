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

# A hard-killed run leaves its remote mcp_app.server holding port 8765 on the
# box; the next session's server then can't bind and every job fast-fails
# ("MCP server unavailable", ~6s, best=None). Sweep our labels' boxes and kill
# stale servers before planning. Best-effort — box may not exist yet.
clean_stale_servers () {
  for DS in $DATASETS; do
    HOST=$("$PY" -c "
import json,sys
try:
    c=json.load(open('eval/eval_config.json'))
    print(c.get('instances',{}).get('$DS-$AUTHOR',{}).get('host',''))
except Exception: print('')
" 2>/dev/null)
    if [ -n "$HOST" ]; then
      echo "@@@ [$(date +%H:%M:%S)] clearing stale mcp_app.server on $DS box $HOST"
      ssh -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
        "ubuntu@$HOST" "pkill -f 'mcp_app.server' || true" 2>/dev/null || true
    fi
  done
}
clean_stale_servers

# Global plan: ALL incomplete defs across BOTH datasets in one cost-sorted
# order, batched into consecutive-same-dataset chunks (bench_fleet is one
# dataset per invocation; a chunk switch costs ~1-2 min of session setup).
# Cheap llama defs interleave with cheap ncnn defs instead of waiting hours.
plan () {  # -> lines: "<dataset> <def> <def> ..."
  "$PY" - "$RESULTS" <<'PYEOF'
import json,sys,glob,re,math
from pathlib import Path
results=Path(sys.argv[1])
items=[]
for p in glob.glob('bench-trace/definitions/**/*.json',recursive=True):
    d=json.load(open(p)); t=d.get('tags',[])
    x=next((s.split(':',1)[1] for s in t if s.startswith('baseline-solution:')),None)
    if x is None and 'simd-loop' in t: x='simd-loop'
    if x in ('ncnn','llama.cpp'): items.append((x,d['name']))
def done(n):
    tp=results/n/'trajectory.jsonl'
    if not tp.exists(): return False
    try: return any(json.loads(l).get('tool')=='submit' for l in open(tp) if l.strip())
    except Exception: return False
def cost(n): return math.prod(max(int(x),1) for x in re.findall(r'\d+',n)) or 1
# Round-robin CAP-sized batches from each dataset, each sorted cheapest-first.
# Pure global cost-sort front-loads all cheap ncnn (poolings cost~1) and pushes
# llama's cheapest (rms_norm cost 2048) hours back — bad when we want to verify
# BOTH harness paths early. Alternating 3-def batches puts llama in chunk 2.
# A switch only costs ~1-2 min (box reused per label).
from collections import defaultdict
CAP=3
byds=defaultdict(list)
for ds,n in sorted([(d,x) for d,x in items if not done(x)], key=lambda t:(cost(t[1]),t[1])):
    byds[ds].append(n)
order=[d for d in ('ncnn','llama.cpp') if byds[d]]
idx={d:0 for d in order}
chunks=[]
while any(idx[d]<len(byds[d]) for d in order):
    for d in order:
        batch=byds[d][idx[d]:idx[d]+CAP]
        if batch: chunks.append((d,batch)); idx[d]+=len(batch)
for ds,ns in chunks: print(ds+' '+' '.join(ns))
PYEOF
}

declare -A PREV
for round in $(seq 1 "$MAX_ROUNDS"); do
  PLAN=$(plan)
  if [ -z "$PLAN" ]; then echo "@@@ ALL COMPLETE at round $round"; break; fi
  # stall detection per dataset: same incomplete count as last round -> fresh box
  for DS in $DATASETS; do
    N=$(echo "$PLAN" | awk -v ds="$DS" '$1==ds {print NF-1}' | paste -sd+ - | bc)
    N=${N:-0}
    if [ "$N" -gt 0 ] && [ "${PREV[$DS]:-}" = "$N" ]; then
      LB="$DS-$AUTHOR"
      echo "@@@ [$(date +%H:%M:%S)] ROUND $round $DS STALLED at $N — tearing down $LB to force a fresh box"
      "$PY" eval/provision.py --teardown --label "$LB" >/dev/null 2>&1 || true
    fi
    PREV[$DS]="$N"
  done
  echo "@@@ [$(date +%H:%M:%S)] ROUND $round plan ($(echo "$PLAN" | wc -l | tr -d ' ') chunks):"
  echo "$PLAN" | sed 's/^/@@@   chunk: /'
  # Read the plan on FD 3, NOT stdin: bench_fleet -> ssh reads stdin and would
  # otherwise swallow the rest of the here-string, so only the first chunk ever
  # ran (llama chunks + heavy ncnn chunks silently skipped). Also give
  # bench_fleet its own </dev/null stdin as belt-and-suspenders.
  while IFS= read -r line <&3; do
    DS=${line%% *}; DEFS=${line#* }
    echo "@@@ [$(date +%H:%M:%S)] chunk $DS ($(echo $DEFS | wc -w | tr -d ' ') defs)"
    "$PY" -u test_scripts/bench_fleet.py --harness claude-code --dataset "$DS" --isa sve2 \
      --model sonnet --author "$AUTHOR" --min-iterations "$MIN_ITERS" --definitions "$DEFS" \
      </dev/null || true
  done 3<<< "$PLAN"
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
