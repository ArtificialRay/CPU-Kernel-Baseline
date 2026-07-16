#!/usr/bin/env bash
# Sequentially run a list of nanobot agent instructions locally, one at a
# time. Each job's stdout/stderr goes straight into its own log file under
# harness_trajs/nanobot/.
#
# Edit the JOBS array below to change what runs. Each entry is "name|prompt".
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/harness_trajs/nanobot"
NANOBOT_DIR="$HOME/l3/CPU-Kernel-Baseline/nanobot"

# ---------------------------------------------------------------------------
# Edit this list. Format: "name|prompt"
# ---------------------------------------------------------------------------
JOBS=(
  "conv2d_ncnn|optimize all 2D convolution kernel at ncnn within 100 iterations. Please dynamically allocate number of iterations and tool calls you spend to optimize one kernel definition, submit your solution if you think the optimization is good enough"
)

mkdir -p "$LOG_DIR"
cd "$NANOBOT_DIR"

for job in "${JOBS[@]}"; do
  name="${job%%|*}"
  prompt="${job#*|}"
  log_file="$LOG_DIR/${name}.log"

  echo "=== [$(date '+%H:%M:%S')] starting job: $name ==="
  nanobot agent -m "$prompt" > "$log_file" 2>&1
  echo "=== [$(date '+%H:%M:%S')] job $name finished -> $log_file ==="
done

echo "All jobs done. Logs in $LOG_DIR"
