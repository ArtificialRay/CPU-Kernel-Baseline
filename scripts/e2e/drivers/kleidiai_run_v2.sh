#!/usr/bin/env bash
# kleidiai_run.sh — the 33rd definition of the teammate's subset, gemm_fp32_n512_k512.
# Runs from ~/arm-bench-kleidiai (a main checkout, because kleidiai harness support exists
# only on main) with main's eval-config change REVERTED to ours (warmup 5, inner_iters 1)
# so these numbers sit alongside the other 32 rather than being measured differently.
#
# An earlier version refused to start while any other lane was provisioning, on the theory
# that terraform state is shared. That was wrong in both directions. The states ARE separate
# (one tfstate per checkout, and provision.py's flock is per-checkout, so it does NOT
# serialise across them) -- but main does not need them to be: the security group is a
# read-only `data` source keyed by var.security_group_id, and key pairs carry a namespace
# suffix, both added precisely so several states can coexist in one account. So there is
# nothing to collide over, at any time. Waiting would never have fixed it either: the SG
# exists as long as any state owns it, so a fresh state would have collided just as hard
# at 3am as at noon.
#
# The real constraint is the 32-vCPU on-demand quota. Each arm is ONE c8g.xlarge (4 vCPU)
# at a time, and we refuse to start an arm that would not fit alongside whatever else is up.
set -u
cd ~/arm-bench-kleidiai
source ~/miniforge3/etc/profile.d/conda.sh; conda activate armbench-run; export WANDB_ENTITY=ArmBench
D=gemm_fp32_n512_k512
QUOTA=32; NEED=4
say(){ echo "[kl $(date -u +%H:%M:%S)] $*"; }

vcpus_in_use(){
  aws ec2 describe-instances --filters "Name=instance-state-name,Values=running,pending" \
    --query "Reservations[].Instances[].InstanceType" --output text 2>/dev/null \
    | tr '\t' '\n' | grep . | awk '
      /\.medium$/ {n+=2; next} /\.large$/ {n+=2; next} /\.xlarge$/ {n+=4; next}
      /\.2xlarge$/{n+=8; next} /\.4xlarge$/{n+=16; next} /\.8xlarge$/{n+=32; next}
      {n+=4} END{print n+0}'
}

wait_for_room(){
  local waited=0 used
  while :; do
    used=$(vcpus_in_use); used=${used:-0}
    if [ $((used + NEED)) -le $QUOTA ]; then say "  ${used}/${QUOTA} vCPU in use; room for this arm"; return 0; fi
    [ $((waited % 600)) -eq 0 ] && say "  ${used}/${QUOTA} vCPU in use, need ${NEED} -- waiting (${waited}s so far)"
    sleep 60; waited=$((waited+60))
    if [ $waited -ge 21600 ]; then say "  gave up waiting for quota after 6h"; return 1; fi
  done
}

run(){ # run <arm> <isa> <min> <max> <group-suffix> [extra...]
  local arm="$1" isa="$2" mn="$3" mx="$4" grp="$5"; shift 5
  wait_for_room || { say "arm $arm SKIPPED (no quota)"; return; }
  say "arm $arm (isa=$isa, $mn/$mx)"
  python test_scripts/bench_fleet.py --harness nanobot --dataset kleidiai --isa "$isa" \
    --instance c8g.xlarge --on-demand --model gpt-5.6-luna \
    --min-iterations "$mn" --max-iterations "$mx" --retries 3 --definitions "$D" \
    --label "kl-$arm" --local-results-dir "/home/ubuntu/arm-bench-kleidiai/agent-runs-kl-$arm" \
    --wandb --wandb-project arm-bench-kernels-gpt5.6-luna --wandb-entity ArmBench \
    --wandb-group "nanobot__gpt-5.6-luna__kleidiai__${grp}" "$@" \
    || say "  arm $arm FAILED (continuing)"
  say "arm $arm done"
}
# NO neon arm. This definition's only expert baseline (baseline-kleidiai-arm) declares
# isa_features: ["sve"] and builds -march=armv8.2-a+sve, so the harness's isa-filter
# drops it under --isa neon ("baseline solution's isa_features aren't satisfied") and
# the run provisions a box only to find no definitions. E1's neon arm is therefore
# structurally 32/33 on the teammate's subset -- not a gap we can close by running more.
# Worth stating in the paper: a vendor-optimized reference does not exist at every ISA
# tier, which bounds what an ISA ablation can compare at all.
run e1sve   sve  40 50  "sve__E1_c8g"
run e1sve2  sve2 40 50  "sve2__S2a"
run s1      sve  100 110 "sve__S1"
run s7seed1 sve  40 50  "sve__S7_seed1" --author nanobot-gpt-5.6-luna-sve-s7seed1
say "kleidiai done -- 33/33 of the subset covered"
