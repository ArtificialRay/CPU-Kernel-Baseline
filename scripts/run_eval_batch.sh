#!/usr/bin/env bash
# Runs eval/run_benchmark.py for a fixed list of op_types, mirroring the
# "arm-bench: run_benchmark" launch.json config (.vscode/launch.json) —
# same python interpreter, dataset, ISA, and model, just looped over --problem.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=/home/rthu/miniconda3/bin/python
ISA=sve
MODEL=openrouter/anthropic/claude-sonnet-4-6

# fp32 definitions — all live under the ncnn dataset
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
    pooling_fp32_global_avg
    pooling_fp32_max_kh2_kw2_sh2_sw2_p0
    pooling_fp32_max_kh3_kw3_sh1_sw1_p1
    pooling_fp32_max_kh3_kw3_sh2_sw2_p0
    pooling_fp32_max_kh3_kw3_sh2_sw2_p1
)

# bf16 definitions — all live under the llama.cpp dataset
LLAMACPP_PROBLEMS=(
    # gemm_bf16_n1024_k2048
    # gemm_bf16_n1408_k2048
    # gemm_bf16_n2048_k1024
    # gemm_bf16_n2048_k1408
    # gemm_bf16_n2048_k2048
    moe_bf16_e60_k4_d2048_ff1408
    moe_bf16_e64_k8_d2048_ff1024
    # mha_bf16_h16_d128_kvh16
    # rms_norm_fp32_d2048
)

run_batch() {
    local dataset="$1"; shift
    for problem in "$@"; do
        echo "=== ${problem} (${dataset}) ==="
        "$PYTHON" -m eval.run_benchmark \
            --problem "$problem" \
            --dataset "$dataset" \
            --isa "$ISA" \
            --model "$MODEL"
    done
}

# run_batch ncnn "${NCNN_PROBLEMS[@]}"
run_batch llama.cpp "${LLAMACPP_PROBLEMS[@]}"
