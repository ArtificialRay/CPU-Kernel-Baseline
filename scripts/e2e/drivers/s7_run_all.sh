#!/usr/bin/env bash
# s7_run_all.sh [first_seed] [last_seed] — queue the S7 seeds, one seed at a time, 3 lanes each.
# Never more than 3 boxes at once (the standing rule), and a seed only starts once the previous
# seed's lanes have all exited, so a stall cannot silently fan out to six instances.
#
# Cost visibility: the OpenAI key has no api.usage.read scope, so spend CANNOT be read from here.
# Between seeds this writes a PAUSE marker and waits for ~/s7_continue_<next> to appear, so the
# run stops at a natural boundary for a human to check platform.openai.com and decide.
# To continue:  touch ~/s7_continue_2   (or _3)
# To stop after the current seed: do nothing.
set -u
L=~/s7logs; mkdir -p $L
FIRST="${1:-1}"; LAST="${2:-3}"
say(){ echo "[s7 $(date -u +%H:%M:%S)] $*"; }

launch_seed() {
  local seed="$1"
  for k in 1 2 3; do
    : > $L/s7_seed${seed}_lane$k.log
    setsid -f bash -c "exec -a s7_s${seed}_l$k bash ~/arm-bench/sweep_logs/s7_lane.sh $seed $k >> $L/s7_seed${seed}_lane$k.log 2>&1 < /dev/null"
    # stagger: wait for this lane's box to exist before starting the next, so provisioning
    # races don't collide in the instance registry
    for i in $(seq 1 90); do grep -aqE 'Instance ready|Reusing existing|terraform apply failed' $L/s7_seed${seed}_lane$k.log && break; sleep 10; done
    sleep 20
  done
  say "seed $seed: 3 lanes launched"
}

for seed in $(seq "$FIRST" "$LAST"); do
  say "=== SEED $seed starting ==="
  launch_seed "$seed"
  sleep 120
  while pgrep -f "s7_s${seed}_l" >/dev/null; do sleep 120; done
  say "=== SEED $seed finished ==="
  n=$(grep -c . /dev/null 2>/dev/null; echo 0)
  next=$((seed + 1))
  [ "$seed" -ge "$LAST" ] && { say "all requested seeds done"; break; }
  say "PAUSE before seed $next — check spend at platform.openai.com, then: touch ~/s7_continue_$next"
  while [ ! -f ~/s7_continue_$next ]; do sleep 60; done
  say "continue marker seen; proceeding to seed $next"
done
say "s7_run_all done"
