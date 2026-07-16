#!/usr/bin/env bash
# Runs mcp_app.smoke_test_driver (the non-nanobot MCP smoke-test driver —
# pushes the reference-scalar kernel through compile/evaluate/disassemble/
# submit over stdio-over-ssh) for a list of problems, per dataset. Mirrors
# run_eval_batch.sh's shape but drives mcp_app.smoke_test_driver instead of
# eval.run_benchmark. Edit NCNN_PROBLEMS/LLAMACPP_PROBLEMS below to change
# which problems get smoke-tested.
#
# Host/user/key are read from eval/eval_config.json (same file
# eval/run_benchmark.py uses) for the instance tier matching ISA — sve ->
# c7g, sve2 -> c8g. Override with HOST/SSH_USER/KEY_FILE env vars, or
# provision one first: python eval/provision.py --isa sve2
#
# Usage:
#   ./scripts/run_driver_smoke.sh
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=/home/rthu/miniconda3/bin/python
ISA=${ISA:-sve}
AUTHOR=${AUTHOR:-driver-smoke-test}
REMOTE_ROOT=${REMOTE_ROOT:-'~/arm-bench'}
SSH_USER=${SSH_USER:-ubuntu}
KEY_FILE=${KEY_FILE:-~/.ssh/id_rsa}

NCNN_PROBLEMS=(
    conv2d_fp32_kh1_kw1_sh1_sw1_dh1_dw1_p0
    conv2d_fp32_kh1_kw1_sh2_sw2_dh1_dw1_p0
    conv2d_fp32_kh3_kw3_sh1_sw1_dh1_dw1_p1
    conv2d_fp32_kh3_kw3_sh2_sw2_dh1_dw1_p1
    conv2d_fp32_kh7_kw7_sh2_sw2_dh1_dw1_p3
    conv2d_depthwise_fp32_kh3_kw3_sh1_sw1_dh1_dw1_p1
    conv2d_depthwise_fp32_kh3_kw3_sh2_sw2_dh1_dw1_p1
    conv2d_depthwise_fp32_kh5_kw5_sh1_sw1_dh1_dw1_p2
    conv2d_depthwise_fp32_kh5_kw5_sh2_sw2_dh1_dw1_p2
    gemm_fp32_n1000_k1280
    gemm_fp32_n1000_k2048
    gemm_fp32_n1280_k960
    gemm_fp32_n29_k800
    #lstm_fp32_i322_h800
    pooling_fp32_global_avg
    pooling_fp32_max_kh2_kw2_sh2_sw2_p0
    pooling_fp32_max_kh3_kw3_sh1_sw1_p1
    pooling_fp32_max_kh3_kw3_sh2_sw2_p0
    pooling_fp32_max_kh3_kw3_sh2_sw2_p1
)
LLAMACPP_PROBLEMS=(
    # gemm_bf16_n1024_k2048
    # gemm_bf16_n1408_k2048
    # gemm_bf16_n2048_k1024
    # gemm_bf16_n2048_k1408
    # gemm_bf16_n2048_k2048
    # mha_bf16_h16_d128_kvh16
    moe_bf16_e60_k4_d2048_ff1408
    moe_bf16_e64_k8_d2048_ff1024
    rms_norm_fp32_d2048
)

# ISA -> eval_config.json instance tier
case "$ISA" in
    sve) TIER=c7g ;;
    sve2) TIER=c8g ;;
    *) TIER="" ;;
esac

if [[ -z "${HOST:-}" ]]; then
    if [[ -z "$TIER" || ! -f eval/eval_config.json ]]; then
        echo "HOST not set and eval/eval_config.json unavailable — set HOST explicitly." >&2
        exit 1
    fi
    HOST=$("$PYTHON" -c "
import json
cfg = json.load(open('eval/eval_config.json'))
print(cfg['instances'].get('$TIER', {}).get('host', ''))
")
    if [[ -z "$HOST" ]]; then
        echo "eval/eval_config.json has no host for tier '$TIER' (ISA=$ISA)." \
             "Provision one first: python eval/provision.py --isa $ISA" >&2
        exit 1
    fi
fi

run_smoke() {
    local dataset=$1 baseline_author=$2
    shift 2
    for problem in "$@"; do
        echo "=== ${problem} (dataset=${dataset}) ==="
        "$PYTHON" -m mcp_app.smoke_test_driver \
            --host "$HOST" --user "$SSH_USER" --key-file "$KEY_FILE" \
            --remote-root "$REMOTE_ROOT" \
            --dataset "$dataset" --baseline-author "$baseline_author" \
            --isa "$ISA" --author "$AUTHOR" \
            --problem "$problem"
    done
}

echo "=== mcp_app.smoke_test_driver smoke test: host=$HOST isa=$ISA author=$AUTHOR ==="
#run_smoke ncnn baseline-ncnn-arm "${NCNN_PROBLEMS[@]}"
run_smoke llama.cpp baseline-llamacpp-arm "${LLAMACPP_PROBLEMS[@]}"
