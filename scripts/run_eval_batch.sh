#!/usr/bin/env bash
# Runs eval/run_benchmark.py for a fixed list of op_types, mirroring the
# "arm-bench: run_benchmark" launch.json config (.vscode/launch.json) —
# same python interpreter, dataset, ISA, and model, just looped over --problem.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=/home/rthu/miniconda3/bin/python
DATASET=ncnn
ISA=sve
MODEL=openrouter/anthropic/claude-sonnet-4-6
PROBLEMS=(
    # pooling_fp32_global_avg
    conv2d_w8a8ch_kh1_kw1_sh1_sw1_dh1_dw1_p0
    conv2d_w8a8ch_kh1_kw1_sh2_sw2_dh1_dw1_p0
    conv2d_w8a8ch_kh3_kw3_sh1_sw1_dh1_dw1_p1
    conv2d_w8a8ch_kh3_kw3_sh2_sw2_dh1_dw1_p1
    conv2d_w8a8ch_kh7_kw7_sh2_sw2_dh1_dw1_p3
    conv2d_depthwise_w8a8ch_kh3_kw3_sh1_sw1_dh1_dw1_p1
    conv2d_depthwise_w8a8ch_kh3_kw3_sh2_sw2_dh1_dw1_p1
    conv2d_depthwise_w8a8ch_kh5_kw5_sh1_sw1_dh1_dw1_p2
    conv2d_depthwise_w8a8ch_kh5_kw5_sh2_sw2_dh1_dw1_p2
    gemm_w8a8ch_n1000_k1280
    gemm_w8a8ch_n1000_k2048
    gemm_w8a8ch_n1280_k960
)

for problem in "${PROBLEMS[@]}"; do
    echo "=== ${problem} ==="
    "$PYTHON" -m eval.run_benchmark \
        --problem "$problem" \
        --dataset "$DATASET" \
        --isa "$ISA" \
        --model "$MODEL"
done
