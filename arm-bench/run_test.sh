#!/bin/bash
# Quick test runner — sources your API key and runs the benchmark
set -e

# Load keys from file if not already set
if [ -f ~/.armbench_key ]; then
    export ANTHROPIC_API_KEY=$(cat ~/.armbench_key | tr -d '[:space:]')
fi
if [ -f ~/.armbench_openrouter_key ]; then
    export OPENROUTER_API_KEY=$(cat ~/.armbench_openrouter_key | tr -d '[:space:]')
fi

PROBLEM="${1:-loop_111}"
ISA="${2:-sve}"
MODEL="${3:-openrouter/anthropic/claude-sonnet-4-6}"

echo "Running: problem=$PROBLEM  isa=$ISA  model=$MODEL"
python -m eval.run_benchmark --problem "$PROBLEM" --isa "$ISA" --model "$MODEL"
