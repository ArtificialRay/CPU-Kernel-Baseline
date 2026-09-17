#!/usr/bin/env bash
# Run ONE lane of a plan with the nanobot harness (sub-experiments S0/S1/S2a/S7...):
#   bash run_lane_nanobot.sh <plan.json> <k> <hours> [model] [suffix]
# Env knobs: LANE_MINIT (min compile+evaluate iterations, default 20 ≈ 40 tool steps),
#   LANE_MAXIT (server evaluate cap, default unset), LANE_ISA (sve), LANE_INSTANCE (c7g.large),
#   LANE_PROJECT (arm-bench-kernels-gpt5.6-luna), LANE_ON_DEMAND=1 for on-demand boxes.
# W&B group per segment: s["wandb_group"] or nanobot__<model>__<dataset>__<isa>[__<suffix>].
# The OpenAI key is never on the command line: the adapter reads NANOBOT_CONFIG_BASE from
# ~/arm-bench/.env, which points at ~/.nanobot/fleet-config.json (chmod 600).
set -uo pipefail
PLAN="${1:?plan.json}"; K="${2:?lane}"; HOURS="${3:-24}"; MODEL="${4:-gpt-5.6-luna}"; SUFFIX="${5:-}"
MINIT="${LANE_MINIT:-20}"; ISA="${LANE_ISA:-sve}"; INSTANCE="${LANE_INSTANCE:-c7g.large}"
PROJECT="${LANE_PROJECT:-arm-bench-kernels-gpt5.6-luna}"
cd "$HOME/arm-bench"
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate armbench-run
export PATH="$HOME/.local/bin:$PATH"
export WANDB_ENTITY=ArmBench
DEADLINE=$(python3 -c "import time; print(time.time() + $HOURS*3600)")
LOG="sweep_logs/nlane$K.log"
echo "=== [$(date '+%F %T')] STATUS-MARKER: nanobot lane $K start (model=$MODEL isa=$ISA minit=$MINIT maxit=${LANE_MAXIT:-none} suffix=${SUFFIX:-none}, deadline in ${HOURS}h) ===" >> "$LOG"
python3 - "$PLAN" "$K" "$MODEL" "$ISA" "$SUFFIX" <<'PY' | while IFS=$'\t' read -r ds label group defs; do
import json, sys
plan = json.load(open(sys.argv[1])); lane = plan["lanes"][int(sys.argv[2]) - 1]
model, isa, suffix = sys.argv[3], sys.argv[4], sys.argv[5]
for s in lane["segments"]:
    label = s.get("label") or f"{s['dataset']}-n{suffix or 'x'}-lane{sys.argv[2]}"
    group = s.get("wandb_group") or (f"nanobot__{model}__{s['dataset']}__{isa}" + (f"__{suffix}" if suffix else ""))
    print("\t".join([s["dataset"], label, group, " ".join(s["definitions"])]))
PY
  echo "=== [$(date '+%F %T')] lane $K: dataset $ds ($(echo $defs | wc -w) kernels) label=$label group=$group ===" >> "$LOG"
  python test_scripts/bench_fleet.py --harness nanobot --dataset "$ds" --isa "$ISA" --instance "$INSTANCE" \
    --model "$MODEL" --min-iterations "$MINIT" ${LANE_MAXIT:+--max-iterations "$LANE_MAXIT"} \
    --until-complete --deadline-epoch "$DEADLINE" \
    --label "$label" --definitions "$defs" --watchdog-minutes 300 \
    --batch-size "$(echo $defs | wc -w)" ${LANE_ON_DEMAND:+--on-demand} \
    --wandb --wandb-project "$PROJECT" --wandb-entity ArmBench \
    --wandb-group "$group" >> "$LOG" 2>&1 < /dev/null
done
echo "=== [$(date '+%F %T')] nanobot lane $K done ===" >> "$LOG"
