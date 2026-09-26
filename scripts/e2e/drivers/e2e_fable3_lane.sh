#!/usr/bin/env bash
# e2e_fable3_lane.sh <lane> — the last two defective kernels, one per lane, under gate v2.
# Completes the set so all 11 are Fable-authored: 6 never had the defect, 5 re-optimized here
# and on 2026-09-22 earlier. Budget 33/42, same as the three that landed at 2.97x/3.03x/3.60x.
set -u
LANE="${1:?lane 1|2}"
cd ~/arm-bench-e2e
source ~/miniforge3/etc/profile.d/conda.sh; conda activate armbench-run
export WANDB_ENTITY=ArmBench
export NANOBOT_CONFIG_BASE=/home/ubuntu/.nanobot/fleet-config.json
export CLAUDE_CODE_OAUTH_TOKEN="$(tr -d '\n' < ~/.claude_oauth_token)"

case "$LANE" in
  1) DEFS="gemm_ggml_q5_K_n8192_k2560";;   # the +1.19 perplexity contributor
  2) DEFS="gemm_ggml_q4_K_n8192_k2560";;   # the +0.22 contributor
  *) echo "bad lane"; exit 1;;
esac

echo "=== [$(date -u +%FT%TZ)] fable3 lane $LANE: $DEFS ==="
python test_scripts/bench_fleet.py --harness claude-code --dataset llama.cpp --isa sve2 \
  --instance c8g.xlarge --on-demand --model claude-fable-5-1 \
  --min-iterations 33 --max-iterations 42 --retries 3 \
  --definitions "$DEFS" --label "e2e-fable3-lane$LANE" \
  --local-results-dir "/home/ubuntu/arm-bench-e2e/agent-runs-e2e-qwen35-4b-fable2" \
  --wandb --wandb-project arm-bench-kernels-gpt5.6-luna --wandb-entity ArmBench \
  --wandb-group "claude-code__claude-fable-5-1__llama.cpp__sve2__E2E_qwen3.5-4b_gatev2"
echo "=== [$(date -u +%FT%TZ)] fable3 lane $LANE done ==="
