#!/usr/bin/env bash
# Sequentially run one nanobot kernel-optimization session per definition,
# one at a time. JOBS is generated (not hand-edited) from every definition
# JSON under bench-trace/definitions/ whose baseline-solution dataset
# matches DATASET below. Each job's stdout/stderr goes straight into its own
# log file under harness_trajs/nanobot/.
#
# DATASET must match whichever dataset the connected MCP server was started
# with (~/.nanobot/config.json's tools.mcpServers entry runs
# `python3 -m mcp_app.server --dataset <dataset> ...`) — a job for a
# definition from a different dataset won't find its resources in that
# session. See skills/nanobot/nanobot-kernel-session/SKILL.md.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/harness_trajs/nanobot"
NANOBOT_DIR="$HOME/l3/CPU-Kernel-Baseline/nanobot"
DEFINITIONS_DIR="$SCRIPT_DIR/bench-trace/definitions"

# ---------------------------------------------------------------------------
# Global knobs — override via env, e.g. `DATASET=simd-loop MAX_ITERATIONS=50 ISA=sve2 ./bench_nanobot_fleet.sh`
# ---------------------------------------------------------------------------
DATASET="${DATASET:-ncnn}"
MAX_ITERATIONS="${MAX_ITERATIONS:-100}"
ISA="${ISA:-sve}"

PROMPT_TEMPLATE='Optimize the "%s" kernel definition (dataset: %s, baseline solution source: %s) in new ISA %s within %s tool calls. Dynamically allocate the number of iterations and tool calls you spend within that budget. Follow the nanobot-kernel-session skill workflow end to end and submit once the optimization is good enough or the iteration budget runs out.'

# ---------------------------------------------------------------------------
# Build JOBS: one "<definition_name>|<prompt>" entry per definition JSON
# under DEFINITIONS_DIR whose baseline-solution dataset (or, for simd-loop
# definitions, whose "simd-loop" tag) matches DATASET. baseline_author
# mirrors the dataset/baseline_author table hand-maintained in SKILL.md §3.
# ---------------------------------------------------------------------------
mapfile -t JOBS < <(python3 - "$DATASET" "$MAX_ITERATIONS" "$DEFINITIONS_DIR" "$PROMPT_TEMPLATE" "$ISA" <<'PYEOF'
import json, sys
from pathlib import Path

dataset, max_iterations, definitions_dir, template, isa = sys.argv[1:6]

BASELINE_AUTHOR_BY_DATASET = {
    "ncnn": "baseline-ncnn-arm",
    "simd-loop": "reference",
    "llama.cpp": "baseline-llamacpp-arm",
}

for path in sorted(Path(definitions_dir).rglob("*.json")):
    d = json.loads(path.read_text())
    tags = d.get("tags", [])
    ds = next((t.split(":", 1)[1] for t in tags if t.startswith("baseline-solution:")), None)
    if ds is None and "simd-loop" in tags:
        ds = "simd-loop"
    if ds != dataset:
        continue
    name = d["name"]
    baseline_author = BASELINE_AUTHOR_BY_DATASET.get(ds, ds)
    prompt = template % (name, ds, baseline_author, isa, max_iterations)
    print(f"{name}|{prompt}")
PYEOF
)

if [ "${#JOBS[@]}" -eq 0 ]; then
  echo "No definitions found for DATASET=$DATASET under $DEFINITIONS_DIR" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
cd "$NANOBOT_DIR"

echo "Running ${#JOBS[@]} job(s) for DATASET=$DATASET, ISA=$ISA, MAX_ITERATIONS=$MAX_ITERATIONS"

for job in "${JOBS[@]}"; do
  name="${job%%|*}"
  prompt="${job#*|}"
  log_file="$LOG_DIR/${DATASET}_${ISA}_${name}.log"

  echo "=== [$(date '+%H:%M:%S')] starting job: $name ==="
  nanobot agent -m "$prompt" > "$log_file" 2>&1
  echo "=== [$(date '+%H:%M:%S')] job $name finished -> $log_file ==="
done

echo "All jobs done. Logs in $LOG_DIR"
