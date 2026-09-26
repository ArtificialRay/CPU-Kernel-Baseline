#!/usr/bin/env bash
# e2e_llama_lane.sh <lane> -- Claude Code + Fable 5.1 on the 7 Llama-3.1-8B packed gemm definitions,
# gate v2 (real activations, calibrated floors). Same protocol as the Qwen3.5-4B set: budget 40/50, sve2, c8g.xlarge.
set -u
LANE="${1:?lane 1|2|3}"
cd ~/arm-bench-e2e
source ~/miniforge3/etc/profile.d/conda.sh; conda activate armbench-run
export WANDB_ENTITY=ArmBench
export NANOBOT_CONFIG_BASE=/home/ubuntu/.nanobot/fleet-config.json
export CLAUDE_CODE_OAUTH_TOKEN="$(tr -d "\n" < ~/.claude_oauth_token)"
case "$LANE" in   # balanced by expected time: lm_head is the slowest to evaluate
  1) DEFS="gemm_ggml_q4_K_n14336_k4096 gemm_ggml_q6_K_n1024_k4096";;
  2) DEFS="gemm_ggml_q6_K_n4096_k14336 gemm_ggml_q4_K_n4096_k4096 gemm_ggml_q4_K_n1024_k4096";;
  3) DEFS="gemm_ggml_q6_K_n128256_k4096 gemm_ggml_q4_K_n4096_k14336";;
  *) echo "bad lane"; exit 1;;
esac
echo "=== [$(date -u +%FT%TZ)] llama lane $LANE: $DEFS ==="
python test_scripts/bench_fleet.py --harness claude-code --dataset llama.cpp --isa sve2 \
  --instance c8g.xlarge --on-demand --model claude-fable-5-1 \
  --min-iterations 40 --max-iterations 50 --retries 3 \
  --definitions "$DEFS" --label "e2e-llama-lane$LANE" \
  --local-results-dir "/home/ubuntu/arm-bench-e2e/agent-runs-e2e-llama8b-fable" \
  --wandb --wandb-project arm-bench-kernels-gpt5.6-luna --wandb-entity ArmBench \
  --wandb-group "claude-code__claude-fable-5-1__llama.cpp__sve2__E2E_llama-3.1-8b"
echo "=== [$(date -u +%FT%TZ)] llama lane $LANE done ==="
