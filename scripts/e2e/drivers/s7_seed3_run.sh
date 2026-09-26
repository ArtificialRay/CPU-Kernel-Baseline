#!/usr/bin/env bash
# s7_seed3_run.sh — run seed 3 WITHOUT waiting for all of seed 2.
#
# s7_run_all.sh serialises seeds: it will not start seed 3 until every seed-2 lane exits.
# Seed 2's lanes 1, 2 and new11 finished hours ago; only lane 3 is still going, so the driver
# is holding 3 idle lanes' worth of work behind one straggler. Quota is 48 vCPU and we are
# using 6. This launches seed 3's lanes 1, 2 and new11 immediately and chains lane 3 to start
# when seed 2's lane 3 exits (so the two lane-3s never overlap, per Allen's framing).
#
# s7_run_all.sh MUST be killed before this runs or it will launch seed 3 a second time.
set -u
L=~/s7logs; N=~/n11logs; mkdir -p $L $N
say(){ echo "[s7s3 $(date -u +%H:%M:%S)] $*"; }

launch_lane(){ # launch_lane <k>
  local k="$1"
  : > $L/s7_seed3_lane$k.log
  setsid -f bash -c "exec -a s7_s3_l$k bash ~/arm-bench/sweep_logs/s7_lane.sh 3 $k >> $L/s7_seed3_lane$k.log 2>&1 < /dev/null"
  for i in $(seq 1 90); do
    grep -aqE 'Instance ready|Reusing existing|terraform apply failed' $L/s7_seed3_lane$k.log && break
    sleep 10
  done
  sleep 20
  say "lane $k launched"
}

say "=== SEED 3 starting (overlapped with seed 2 lane 3) ==="
launch_lane 1
launch_lane 2

say "launching new11 lane for seed 3"
: > $N/s7_seed3.log
setsid -f bash -c "exec -a n11_s7s3 bash ~/arm-bench/sweep_logs/new11_lane.sh s7 sve 3 >> $N/s7_seed3.log 2>&1 < /dev/null"
for i in $(seq 1 90); do
  grep -aqE 'Instance ready|Reusing existing|terraform apply failed' $N/s7_seed3.log && break
  sleep 10
done
say "new11 lane launched"

say "waiting for seed 2 lane 3 to finish before starting seed 3 lane 3"
while pgrep -f "s7_s2_l3" >/dev/null; do sleep 60; done
say "seed 2 lane 3 done"
launch_lane 3

say "all seed 3 lanes running; waiting for completion"
sleep 120
while pgrep -f "s7_s3_l" >/dev/null || pgrep -f "n11_s7s3" >/dev/null; do sleep 120; done
say "=== SEED 3 finished === S7 is now n=3"
