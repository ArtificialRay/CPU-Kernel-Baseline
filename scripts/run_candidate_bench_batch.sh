#!/usr/bin/env bash
# Local counterpart to run_eval_batch.sh: instead of driving the agentic SSH
# eval (Path 1, eval/run_benchmark.py), this SSHes into the provisioned
# Graviton instance and runs bench.cli directly there (Path 2, no local
# clang++/ncnn/llama.cpp build needed). Host/user/key come from
# eval/eval_config.json, same as sync_remote.sh. Each `bench.cli bench` call
# writes its trace(s) to the remote ~/arm-bench/bench-trace/traces/
# warehouse — pull them back with sync_remote.sh (reverse) or an explicit
# rsync/scp if you need them locally.
#
# Usage:
#   ./scripts/run_bench_batch.sh                # uses tier from TIER (default c7g)
#   TIER=c8g ./scripts/run_bench_batch.sh        # SVE2 instance
#   HOST=1.2.3.4 ./scripts/run_bench_batch.sh    # override host directly
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$REPO_ROOT/eval/eval_config.json"
TIER="${TIER:-c7g}"
KEY="${KEY:-$HOME/.ssh/id_rsa}"
USER_NAME="${USER_NAME:-ubuntu}"
AUTHOR=reference-scalar

if [[ -z "${HOST:-}" ]]; then
    if [[ ! -f "$CONFIG" ]]; then
        echo "error: $CONFIG not found and HOST env not set" >&2
        exit 1
    fi
    HOST=$(python3 -c "import json; print(json.load(open('$CONFIG'))['instances']['$TIER']['host'])")
fi
if [[ -z "$HOST" ]]; then
    echo "error: empty host for tier '$TIER' (instance not provisioned?)" >&2
    exit 1
fi

SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$USER_NAME@$HOST")

# fp32 definitions — all live under the ncnn dataset
NCNN_DEFS=(
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
    pooling_fp32_global_avg
    pooling_fp32_max_kh2_kw2_sh2_sw2_p0
    pooling_fp32_max_kh3_kw3_sh1_sw1_p1
    pooling_fp32_max_kh3_kw3_sh2_sw2_p0
    pooling_fp32_max_kh3_kw3_sh2_sw2_p1
)

# bf16 definitions — all live under the llama.cpp dataset
LLAMACPP_DEFS=(
    gemm_bf16_n1024_k2048
    gemm_bf16_n1408_k2048
    gemm_bf16_n2048_k1024
    gemm_bf16_n2048_k1408
    gemm_bf16_n2048_k2048
    # moe_bf16_e60_k4_d2048_ff1408
    # moe_bf16_e64_k8_d2048_ff1024
    mha_bf16_h16_d128_kvh16
    # rms_norm_bf16_d2048
)

run_batch() {
    local baseline_author="$1"; shift
    for def_name in "$@"; do
        echo "=== ${def_name} (${USER_NAME}@${HOST}) ==="
        "${SSH[@]}" "cd ~/arm-bench && python3 -m bench.cli bench --definition '${def_name}' --solution '${AUTHOR}_${def_name}' --baseline-author '${baseline_author}'"
    done
}

# run_batch baseline-ncnn-arm "${NCNN_DEFS[@]}"
run_batch baseline-llamacpp-arm "${LLAMACPP_DEFS[@]}"
