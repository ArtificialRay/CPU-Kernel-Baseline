#!/usr/bin/env bash
# Sequentially run one nanobot kernel-optimization session per definition,
# one at a time. JOBS is generated (not hand-edited) from every definition
# JSON under bench-trace/definitions/ whose baseline-solution dataset
# matches DATASET below (optionally narrowed to DEFINITIONS). Each job's
# stdout/stderr goes straight into its own log file under harness_trajs/nanobot/.
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
# Global knobs — override via env, e.g. `DATASET=simd-loop MAX_ITERATIONS=50 ISA=sve2 ./bench_nanobot_recover.sh`
#
# DEFINITIONS: optional allow-list of definitions to recover (e.g. the ones
# that failed with a dead MCP connection). Leave unset to run every
# definition under DATASET, same as bench_nanobot_fleet.sh. Accepts either:
#   - space-separated bare names:
#       DEFINITIONS="gemm_bf16_n1408_k2048 gemm_bf16_n2048_k1024" ./bench_nanobot_recover.sh
#   - a JSON array, bare names or full "<dataset>_<isa>_<name>" log-file
#     stems (as copy-pasted straight out of harness_trajs/nanobot/*.log
#     filenames) — the dataset_isa_ prefix is stripped automatically:
#       DEFINITIONS='[
#         "llama.cpp_sve_moe_bf16_e64_k8_d2048_ff1024",
#         "llama.cpp_sve_moe_q4_k_m_e60_k4_d2048_ff1536"
#       ]' DATASET=llama.cpp ISA=sve ./bench_nanobot_recover.sh
# ---------------------------------------------------------------------------
DATASET="${DATASET:-ncnn}"
MAX_ITERATIONS="${MAX_ITERATIONS:-100}"
ISA="${ISA:-sve}"
DEFINITIONS='
[
"llama.cpp_sve_moe_q4_k_m_e60_k4_d2048_ff1536",
"llama.cpp_sve_moe_q4_k_m_e64_k8_d2048_ff1024",
"llama.cpp_sve_moe_q8_0_e60_k4_d2048_ff1408",
"llama.cpp_sve_moe_q8_0_e64_k8_d2048_ff1024",
"llama.cpp_sve_rms_norm_fp32_d2048"
]
'

PROMPT_TEMPLATE='Optimize the "%s" kernel definition (dataset: %s, baseline solution source: %s) in new ISA %s within %s tool calls. Dynamically allocate the number of iterations and tool calls you spend within that budget. Follow the nanobot-kernel-session skill workflow end to end and submit once the optimization is good enough or the iteration budget runs out.'

# ---------------------------------------------------------------------------
# Build JOBS: one "<definition_name>|<prompt>" entry per definition JSON
# under DEFINITIONS_DIR whose baseline-solution dataset (or, for simd-loop
# definitions, whose "simd-loop" tag) matches DATASET. baseline_author
# mirrors the dataset/baseline_author table hand-maintained in SKILL.md §3.
# ---------------------------------------------------------------------------
mapfile -t JOBS < <(python3 - "$DATASET" "$MAX_ITERATIONS" "$DEFINITIONS_DIR" "$PROMPT_TEMPLATE" "$ISA" "$DEFINITIONS" <<'PYEOF'
import json, sys
from pathlib import Path

dataset, max_iterations, definitions_dir, template, isa, definitions_filter = sys.argv[1:7]

BASELINE_AUTHOR_BY_DATASET = {
    "ncnn": "baseline-ncnn-arm",
    "simd-loop": "reference",
    "llama.cpp": "baseline-llamacpp-arm",
}

definitions_filter = definitions_filter.strip()
if definitions_filter.startswith("["):
    raw_entries = json.loads(definitions_filter)
else:
    raw_entries = definitions_filter.split()

log_stem_prefix = f"{dataset}_{isa}_"
wanted = {
    entry[len(log_stem_prefix):] if entry.startswith(log_stem_prefix) else entry
    for entry in raw_entries
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
    if wanted and name not in wanted:
        continue
    baseline_author = BASELINE_AUTHOR_BY_DATASET.get(ds, ds)
    prompt = template % (name, ds, baseline_author, isa, max_iterations)
    print(f"{name}|{prompt}")
PYEOF
)

if [ "${#JOBS[@]}" -eq 0 ]; then
  echo "No definitions found for DATASET=$DATASET (DEFINITIONS=${DEFINITIONS:-<all>}) under $DEFINITIONS_DIR" >&2
  exit 1
fi

if [ -n "$DEFINITIONS" ]; then
  requested_count=$(python3 -c '
import json, sys
s = sys.argv[1].strip()
entries = json.loads(s) if s.startswith("[") else s.split()
print(len(entries))
' "$DEFINITIONS")
  if [ "${#JOBS[@]}" -ne "$requested_count" ]; then
    echo "WARNING: requested $requested_count definition(s) via DEFINITIONS, but only found ${#JOBS[@]} matching DATASET=$DATASET." >&2
  fi
fi

mkdir -p "$LOG_DIR"
cd "$NANOBOT_DIR"

echo "Running ${#JOBS[@]} job(s) for DATASET=$DATASET, ISA=$ISA, MAX_ITERATIONS=$MAX_ITERATIONS${DEFINITIONS:+, DEFINITIONS=$DEFINITIONS}"

for job in "${JOBS[@]}"; do
  name="${job%%|*}"
  prompt="${job#*|}"
  log_file="$LOG_DIR/${DATASET}_${ISA}_${name}.log"

  echo "=== [$(date '+%H:%M:%S')] starting job: $name ==="
  nanobot agent --logs -m "$prompt" > "$log_file" 2>&1  --session "$(date +%Y%m%d-%H%M%S)"
  echo "=== [$(date '+%H:%M:%S')] job $name finished -> $log_file ==="
done

echo "All jobs done. Logs in $LOG_DIR"
