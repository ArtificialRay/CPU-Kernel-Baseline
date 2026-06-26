#!/usr/bin/env bash
# Overnight agent timing sweep.
#
# Drives scripts/bench_loop_agent.py (the in-process LLM optimization agent)
# across many simd-loops, repeating each loop for timing stability, and captures
# a per-run log plus a summary CSV. Designed to run unattended in tmux on the
# Graviton4 target (real SVE2 timing). Resilient: a failure on one loop/repeat
# is logged and the sweep continues.
#
# Requirements: OPENROUTER_API_KEY in the environment (the agent calls litellm).
#
# Usage (on the Graviton instance, inside ~/arm-bench):
#   OPENROUTER_API_KEY=sk-or-... tmux new-session -d -s agent \
#     "cd ~/arm-bench && bash scripts/overnight_agent_sweep.sh 2>&1 | tee ~/agent_sweep.log"
#
# Tunables (env vars):
#   LOOPS    space-separated loop ids   (default: a cross-shape set; the agent
#            now supports ALL 47 loops — scalar, array-output, and sort — via the
#            standard runner/evaluator, so any loop_id is valid here)
#   TURNS    agent turns per loop        (default: 5)
#   REPEATS  passes over the loop set    (default: 3)
#   MODEL    litellm model id            (default: openrouter/anthropic/claude-opus-4-6)
#   OUTDIR   output directory            (default: ~/agent_runs/<timestamp>)
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

LOOPS="${LOOPS:-loop_001 loop_002 loop_004 loop_024 loop_028 loop_035 loop_108 loop_113 loop_120 loop_123 loop_217}"
TURNS="${TURNS:-5}"
REPEATS="${REPEATS:-3}"
MODEL="${MODEL:-openrouter/anthropic/claude-opus-4-6}"
OUTDIR="${OUTDIR:-$HOME/agent_runs/$(date +%Y%m%d_%H%M%S)}"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "ERROR: OPENROUTER_API_KEY is not set" >&2
  exit 1
fi

mkdir -p "$OUTDIR"
SUMMARY="$OUTDIR/summary.csv"
echo "timestamp,loop,repeat,status,best_ns,logfile" > "$SUMMARY"

echo "[overnight] start $(date -u)  host=$(hostname)  arch=$(uname -m)"
echo "[overnight] model=$MODEL turns=$TURNS repeats=$REPEATS"
echo "[overnight] loops: $LOOPS"
echo "[overnight] outdir: $OUTDIR"
echo

n_loops=$(wc -w <<< "$LOOPS")
total=$(( n_loops * REPEATS ))
done=0

for rep in $(seq 1 "$REPEATS"); do
  for loop in $LOOPS; do
    done=$(( done + 1 ))
    log="$OUTDIR/${loop}_rep${rep}.log"
    echo "[overnight] $(date -u +%H:%M:%S) [$done/$total] $loop rep $rep/$REPEATS -> $log"
    if python3 scripts/bench_loop_agent.py --loop "$loop" --max-turns "$TURNS" \
         --model "$MODEL" --key "$OPENROUTER_API_KEY" > "$log" 2>&1; then
      status=ok
    else
      status=fail
    fi
    best=$(grep -oE 'Best timing: [0-9]+ ns' "$log" | tail -1 | grep -oE '[0-9]+' || true)
    [[ -z "$best" ]] && best=NA
    echo "         -> status=$status best_ns=$best"
    echo "$(date -u +%FT%TZ),$loop,$rep,$status,$best,$log" >> "$SUMMARY"
  done
done

echo
echo "[overnight] DONE $(date -u)"
echo "[overnight] summary ($SUMMARY):"
column -t -s, "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
