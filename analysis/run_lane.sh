#!/usr/bin/env bash
# Run ONE lane of the plan: bash run_lane.sh <plan.json> <k> <hours>
set -uo pipefail
PLAN="${1:?plan.json}"; K="${2:?lane}"; HOURS="${3:-24}"
cd "$HOME/arm-bench"
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate armbench-run
export PATH="$HOME/.local/bin:$PATH"
export CLAUDE_CODE_OAUTH_TOKEN="$(tr -d '[:space:]' < "$HOME/.claude_oauth_token")"
export WANDB_ENTITY=ArmBench
DEADLINE=$(python3 -c "import time; print(time.time() + $HOURS*3600)")
LOG="sweep_logs/lane$K.log"
echo "=== [$(date '+%F %T')] STATUS-MARKER: lane $K start (deadline in ${HOURS}h) ===" >> "$LOG"
python3 - "$PLAN" "$K" <<'PY' | while IFS=$'\t' read -r ds defs; do
import json, sys
plan = json.load(open(sys.argv[1])); lane = plan["lanes"][int(sys.argv[2]) - 1]
for s in lane["segments"]: print(s["dataset"] + "\t" + " ".join(s["definitions"]))
PY
  echo "=== [$(date '+%F %T')] lane $K: dataset $ds ($(echo $defs | wc -w) kernels) ===" >> "$LOG"
  python test_scripts/bench_fleet.py --harness claude-code --dataset "$ds" --isa sve --instance c7g.large \
    --model claude-sonnet-4-6 --min-iterations 40 --until-complete --deadline-epoch "$DEADLINE" \
    --label "$ds-lane$K" --definitions "$defs" --watchdog-minutes 300 \
    --wandb --wandb-project arm-bench-kernels --wandb-entity ArmBench \
    --wandb-group "claude-code__claude-sonnet-4-6__${ds}__sve" >> "$LOG" 2>&1 < /dev/null
done
echo "=== [$(date '+%F %T')] lane $K done ===" >> "$LOG"
