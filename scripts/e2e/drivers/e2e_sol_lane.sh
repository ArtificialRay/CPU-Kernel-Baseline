#!/usr/bin/env bash
# e2e_sol_lane.sh <lane> — E2E Qwen3.5-4B kernels, nanobot + gpt-5.6-sol (OpenAI key),
# run under the FIXED gate (real activations + realistic k-quant weights + baseline-relative
# SQNR floor, commit ebf4a29). Same fleet driver and box class as the Fable run so the two
# are comparable; only the model and the gate differ.
#
# Lane 1 = all 11 definitions ordered by decode byte share. Lanes 2/3 split them for a
# 3-box expansion (launch lane 1 with LANES3=1 to use its 4-definition subset instead).
set -u
LANE="${1:?lane 1|2|3}"
cd ~/arm-bench-e2e
source ~/miniforge3/etc/profile.d/conda.sh; conda activate armbench-run
export WANDB_ENTITY=ArmBench
# nanobot must read the fleet config (OpenAI provider + key), not the repo's key-less config.json
export NANOBOT_CONFIG_BASE=/home/ubuntu/.nanobot/fleet-config.json

ALL="gemm_ggml_q4_K_n9216_k2560 gemm_ggml_q6_K_n2560_k9216 gemm_ggml_q5_K_n8192_k2560 \
gemm_ggml_q4_K_n2560_k9216 gemm_ggml_q6_K_n248320_k2560 gemm_ggml_q4_K_n8192_k2560 \
gemm_ggml_q4_K_n4096_k2560 gemm_ggml_q5_K_n2560_k4096 gemm_ggml_q4_K_n2560_k4096 \
gemm_ggml_q6_K_n1024_k2560 gemm_ggml_q4_K_n1024_k2560"

case "$LANE" in
  1) DEFS="${LANES3:+gemm_ggml_q4_K_n9216_k2560 gemm_ggml_q6_K_n2560_k9216 gemm_ggml_q5_K_n8192_k2560 gemm_ggml_q4_K_n2560_k9216}"
     DEFS="${DEFS:-$ALL}";;
  2) DEFS="gemm_ggml_q6_K_n248320_k2560 gemm_ggml_q4_K_n8192_k2560 gemm_ggml_q4_K_n4096_k2560 gemm_ggml_q5_K_n2560_k4096";;
  3) DEFS="gemm_ggml_q4_K_n2560_k4096 gemm_ggml_q6_K_n1024_k2560 gemm_ggml_q4_K_n1024_k2560";;
  *) echo "bad lane"; exit 1;;
esac

echo "=== [$(date -u +%FT%TZ)] sol lane $LANE: $(echo $DEFS | wc -w) definitions ==="
python test_scripts/bench_fleet.py --harness nanobot --dataset llama.cpp --isa sve2 \
  --instance c8g.xlarge --on-demand --model gpt-5.6-sol \
  --min-iterations "${MINIT:-50}" --max-iterations "${MAXIT:-60}" --retries 3 \
  --definitions "$DEFS" --label "e2e-sol-lane$LANE" \
  --local-results-dir "/home/ubuntu/arm-bench-e2e/agent-runs-e2e-qwen35-4b-sol" \
  --wandb --wandb-project arm-bench-kernels-gpt5.6-luna --wandb-entity ArmBench \
  --wandb-group "nanobot__gpt-5.6-sol__llama.cpp__sve2__E2E_qwen3.5-4b_gatev2"
echo "=== [$(date -u +%FT%TZ)] sol lane $LANE done ==="
