#!/usr/bin/env bash
# Runs eval/run_benchmark.py (own harness) for a fixed list of definitions,
# one --problem call each, reusing whatever instance is already up for
# DATASET+ISA (no --provision/--teardown).
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-/home/rthu/miniconda3/bin/python}"
DATASET="${DATASET:-ncnn}"
ISA="${ISA:-sve}"
MODEL="${MODEL:-openrouter/anthropic/claude-sonnet-4-6}"
# Set ARMBENCH_LABEL_SUFFIX to avoid colliding with another run on the same
# DATASET+ISA instance (see eval/run_benchmark.py's _label_for).

# Edit this list — one definition name or op_type prefix per run.
DEFINITIONS=(
    # pooling_fp32_global_avg
    lstm_fp32_i322_h800
)

for problem in "${DEFINITIONS[@]}"; do
    echo "=== ${problem} ==="
    "$PYTHON" -m eval.run_benchmark \
        --problem "$problem" \
        --dataset "$DATASET" \
        --isa "$ISA" \
        --model "$MODEL"
done
