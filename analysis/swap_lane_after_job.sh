#!/usr/bin/env bash
# swap_lane_after_job.sh <k> <new_plan.json> [hours]
# Wait until lane <k> finishes the kernel it is on now (its next "job ... finished"
# marker, then that job's W&B upload), SIGKILL the lane tree (no SIGTERM, so no
# scoped teardown — the box stays for reuse) and relaunch the lane on <new_plan>.
set -uo pipefail
K="${1:?lane}"; PLAN="${2:?plan.json}"; HOURS="${3:-40}"; MODEL="${4:-claude-sonnet-4-6}"
cd "$HOME/arm-bench"; LOG="sweep_logs/lane$K.log"
n0=$(wc -l < "$LOG")
until tail -n +"$n0" "$LOG" | grep -qaE "^=== \[[0-9:]+\] job \S+ finished"; do sleep 10; done
# give the just-finished job's W&B sync time to complete (wandb prints "Find logs at" last)
for _ in $(seq 1 30); do tail -n +"$n0" "$LOG" | grep -qa "wandb: Find logs at" && break; sleep 5; done
desc() { for c in $(pgrep -P "$1"); do desc "$c"; done; echo "$1"; }
pids=""; for root in $(pgrep -f "^run_lane_$K "); do pids="$pids $(desc $root)"; done
echo "=== [$(date '+%F %T')] SWAP: lane $K finished its kernel; killing tree ($pids) and relaunching on $PLAN ===" >> "$LOG"
for p in $pids; do kill -KILL "$p" 2>/dev/null; done; sleep 3
setsid -f bash -c "exec -a run_lane_$K bash run_lane.sh $PLAN $K $HOURS $MODEL" > /dev/null 2>&1 < /dev/null
echo "=== [$(date '+%F %T')] SWAP: lane $K relaunched ===" >> "$LOG"
