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
PROBLEMS=(pooling_fp32_global_avg)

for problem in "${PROBLEMS[@]}"; do
    echo "=== ${problem} ==="
    "$PYTHON" -m eval.run_benchmark \
        --problem "$problem" \
        --dataset "$DATASET" \
        --isa "$ISA" \
        --model "$MODEL"
done
