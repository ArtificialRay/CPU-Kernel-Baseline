#!/usr/bin/env bash
# Run ONE lane of the plan: bash run_lane.sh <plan.json> <k> <hours>
set -uo pipefail
PLAN="${1:?plan.json}"; K="${2:?lane}"; HOURS="${3:-24}"; MODEL="${4:-claude-sonnet-4-6}"
cd "$HOME/arm-bench"
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate armbench-run
export PATH="$HOME/.local/bin:$PATH"
export CLAUDE_CODE_OAUTH_TOKEN="$(tr -d '[:space:]' < "$HOME/.claude_oauth_token")"
export WANDB_ENTITY=ArmBench
DEADLINE=$(python3 -c "import time; print(time.time() + $HOURS*3600)")
LOG="sweep_logs/lane$K.log"
echo "=== [$(date '+%F %T')] STATUS-MARKER: lane $K start (deadline in ${HOURS}h) ===" >> "$LOG"
# A segment may override its box label and W&B group (e.g. a validation set that
# must not land in the protocol group): {"dataset","definitions","label"?,"wandb_group"?}
python3 - "$PLAN" "$K" "$MODEL" <<'PY' | while IFS=$'\t' read -r ds label group defs; do
import json, sys
plan = json.load(open(sys.argv[1])); lane = plan["lanes"][int(sys.argv[2]) - 1]
for s in lane["segments"]:
    label = s.get("label") or f"{s['dataset']}-lane{sys.argv[2]}"
    group = s.get("wandb_group") or f"claude-code__{sys.argv[3]}__{s['dataset']}__sve"
    print("\t".join([s["dataset"], label, group, " ".join(s["definitions"])]))
PY
  echo "=== [$(date '+%F %T')] lane $K: dataset $ds ($(echo $defs | wc -w) kernels) label=$label group=$group ===" >> "$LOG"
  python test_scripts/bench_fleet.py --harness claude-code --dataset "$ds" --isa sve --instance c7g.large \
    --model "$MODEL" --min-iterations 40 --until-complete --deadline-epoch "$DEADLINE" \
    --label "$label" --definitions "$defs" --watchdog-minutes 300 \
    --batch-size "$(echo $defs | wc -w)" ${LANE_ON_DEMAND:+--on-demand} \
    --wandb --wandb-project arm-bench-kernels --wandb-entity ArmBench \
    --wandb-group "$group" >> "$LOG" 2>&1 < /dev/null
done
echo "=== [$(date '+%F %T')] lane $K done ===" >> "$LOG"
