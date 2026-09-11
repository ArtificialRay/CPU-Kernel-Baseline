#!/usr/bin/env bash
# Launch every lane of the plan: bash run_lanes.sh <plan.json> <hours>
# Each lane runs via run_lane.sh (segments sequential, own --label per dataset). Re-runnable.
set -uo pipefail
PLAN="${1:?plan.json}"; HOURS="${2:-24}"
cd "$HOME/arm-bench"
pkill -f "[r]un_lane_[0-9]" 2>/dev/null; sleep 1; pkill -f "^\S*python\S* \S*bench_fleet.py" 2>/dev/null; sleep 1
pkill -f "[c]laude -p" 2>/dev/null; pkill -f "[s]sh -L" 2>/dev/null; sleep 2
NL=$(python3 -c "import json; print(len(json.load(open('$PLAN'))['lanes']))")
for k in $(seq 1 "$NL"); do
  setsid -f bash -c "exec -a run_lane_$k bash run_lane.sh $PLAN $k $HOURS" > /dev/null 2>&1 < /dev/null
  [ "$k" -lt "$NL" ] && sleep 150   # stagger: terraform lock + eval_config writes
done
echo "lanes launched: $(pgrep -f '[r]un_lane_[0-9]' | wc -l)"
