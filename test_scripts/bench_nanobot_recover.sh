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

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_DIR/harness_trajs/nanobot"
NANOBOT_DIR="$HOME/l3/CPU-Kernel-Baseline/nanobot"
DEFINITIONS_DIR="$REPO_DIR/bench-trace/definitions"
GLOBAL_WORKSPACE="${NANOBOT_WORKSPACE:-$HOME/.nanobot/workspace}"
JOB_WORKSPACES_DIR="$HOME/.nanobot/job_workspaces"

if [ ! -d "$GLOBAL_WORKSPACE" ]; then
  echo "GLOBAL_WORKSPACE ($GLOBAL_WORKSPACE) doesn't exist yet — run" \
       "'nanobot agent -m \"hi\"' once to bootstrap AGENTS.md/SOUL.md/skills/ before using this script." >&2
  exit 1
fi

make_job_workspace() {
  local job_ws="$JOB_WORKSPACES_DIR/$1"
  mkdir -p "$job_ws"
  local shared
  # Copied (not symlinked) into the job workspace to fit sandboxing requirements
  for shared in AGENTS.md HEARTBEAT.md SOUL.md USER.md prompts skills; do
    if [ -e "$GLOBAL_WORKSPACE/$shared" ]; then
      rsync -a --delete --exclude='.git' "$GLOBAL_WORKSPACE/$shared" "$job_ws/"
    fi
  done
  echo "$job_ws"
}

# Best-effort: a sync failure (or an unreachable/reclaimed instance) is
# logged and swallowed, never fatal — losing the ability to sync one job's
# results shouldn't abort the rest of the batch.
sync_job_results() {
  local definition="$1"
  if [ ! -f "$EVAL_CONFIG" ]; then
    echo "  WARNING: $EVAL_CONFIG not found — skipping sync-results for $definition" >&2
    return 0
  fi
  local host user key_file
  host="$(python3 -c "
import json
print(json.load(open('$EVAL_CONFIG'))['instances'].get('$LABEL', {}).get('host', ''))
")"
  if [ -z "$host" ]; then
    echo "  WARNING: no host recorded for label=$LABEL in $EVAL_CONFIG — skipping sync-results for $definition" >&2
    return 0
  fi
  user="$(python3 -c "
import json
print(json.load(open('$EVAL_CONFIG'))['instances']['$LABEL'].get('user', 'ubuntu'))
")"
  key_file="$(python3 -c "
import json
print(json.load(open('$EVAL_CONFIG'))['instances']['$LABEL'].get('key_file', '~/.ssh/id_rsa'))
")"
  python3 "$REPO_DIR/skills/launch/launch_session.py" sync-results \
    --host "$host" --user "$user" --key-file "$key_file" \
    --author "$AUTHOR" --definition "$definition" \
    --local-results-dir "$LOCAL_RESULTS_DIR" \
    || echo "  WARNING: sync-results failed for $definition" >&2
}

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
NANOBOT_CONFIG="$REPO_DIR/skills/nanobot/nanobot-kernel-session/config.json"
# Defensive backstop against transient infra failures — see
# bench_nanobot_fleet.sh's RETRIES comment for the full rationale. Total
# attempts = RETRIES+1.
RETRIES="${RETRIES:-3}"

# Sync each job's results back to the local checkout right after it finishes
# (skills/launch/launch_session.py sync-results — pulls
# agent-runs-mcp/<author>/<definition>/ from the remote instance), so
# already-finished jobs' results survive even if a later job's connection
# dies mid-batch (e.g. AWS reclaiming a spot instance — see
# skills/README.md's "After the run: sync results back"). AUTHOR must match
# whatever --author the MCP server this session connects to was launched
# with (default "nanobot" everywhere in this repo). HOST/USER/KEY_FILE are
# read from eval/eval_config.json, keyed by LABEL — must match whatever
# --label the instance serving this DATASET+ISA was launched/provisioned
# under (default f"{DATASET}-{ISA}", mirroring eval/provision.py's
# default_label() — see its module docstring). Override LABEL explicitly if
# you launched with a custom --label.
AUTHOR="${AUTHOR:-nanobot}"
LOCAL_RESULTS_DIR="${LOCAL_RESULTS_DIR:-$REPO_DIR/agent-runs-$AUTHOR}"
EVAL_CONFIG="$REPO_DIR/eval/eval_config.json"
LABEL="${LABEL:-${DATASET}-${ISA}}"

DEFINITIONS='
[
"conv2d_fp32_kh3_kw3_sh1_sw1_dh1_dw1_p1",
"conv2d_fp32_kh3_kw3_sh2_sw2_dh1_dw1_p1",
"conv2d_fp32_kh7_kw7_sh2_sw2_dh1_dw1_p3",
"conv2d_w8a8ch_kh1_kw1_sh1_sw1_dh1_dw1_p0",
"conv2d_w8a8ch_kh1_kw1_sh2_sw2_dh1_dw1_p0",
"conv2d_w8a8ch_kh3_kw3_sh1_sw1_dh1_dw1_p1",
"conv2d_w8a8ch_kh3_kw3_sh2_sw2_dh1_dw1_p1",
"conv2d_w8a8ch_kh7_kw7_sh2_sw2_dh1_dw1_p3",
"conv2d_depthwise_fp32_kh3_kw3_sh1_sw1_dh1_dw1_p1",
"conv2d_depthwise_fp32_kh3_kw3_sh2_sw2_dh1_dw1_p1",
"conv2d_depthwise_fp32_kh5_kw5_sh1_sw1_dh1_dw1_p2",
"conv2d_depthwise_fp32_kh5_kw5_sh2_sw2_dh1_dw1_p2",
"conv2d_depthwise_w8a8ch_kh3_kw3_sh1_sw1_dh1_dw1_p1",
"conv2d_depthwise_w8a8ch_kh3_kw3_sh2_sw2_dh1_dw1_p1",
"conv2d_depthwise_w8a8ch_kh5_kw5_sh1_sw1_dh1_dw1_p2",
"conv2d_depthwise_w8a8ch_kh5_kw5_sh2_sw2_dh1_dw1_p2",
"gemm_fp32_n1000_k1280",
"gemm_fp32_n1000_k2048",
"gemm_fp32_n1280_k960",
"gemm_fp32_n29_k800",
"gemm_w8a8ch_n1000_k1280",
"gemm_w8a8ch_n1000_k2048",
"gemm_w8a8ch_n1280_k960",
"pooling_fp32_global_avg",
"pooling_fp32_max_kh2_kw2_sh2_sw2_p0",
"pooling_fp32_max_kh3_kw3_sh1_sw1_p1",
"pooling_fp32_max_kh3_kw3_sh2_sw2_p0",
"pooling_fp32_max_kh3_kw3_sh2_sw2_p1"
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
  job_workspace="$(make_job_workspace "${DATASET}_${ISA}_${name}")"

  attempt=0
  while true; do
    echo "=== [$(date '+%H:%M:%S')] starting job: $name (workspace: $job_workspace) (attempt $((attempt + 1))/$((RETRIES + 1))) ==="
    set +e
    nanobot agent --logs -m "$prompt" -w "$job_workspace" -c "$NANOBOT_CONFIG" --session "$(date +%Y%m%d-%H%M%S)" > "$log_file" 2>&1
    rc=$?
    set -e
    if [ "$rc" -eq 0 ]; then
      break
    fi
    # Known benign case: nanobot's close_mcp() can crash on asyncio.CancelledError
    # while tearing down two concurrently-connected MCP sessions (only reproduces
    # when both NCNNKernelBench and LLAMACPPKernelBench are simultaneously
    # reachable — i.e. running two datasets' fleets in parallel against two live
    # instances). It happens AFTER the job's own work is done — evaluate() already
    # auto-persisted the best version remotely — so it's safe to treat as a warning
    # and move on (no retry needed) rather than letting `set -e` abort the whole batch.
    if grep -q "asyncio.exceptions.CancelledError" "$log_file" && grep -q "close_mcp" "$log_file"; then
      echo "  WARNING: job $name crashed during MCP cleanup after finishing (known nanobot close_mcp() CancelledError bug) — result may already be persisted remotely, continuing" >&2
      break
    fi
    if [ "$attempt" -ge "$RETRIES" ]; then
      echo "  ERROR: job $name's nanobot process exited $rc after $((attempt + 1)) attempt(s) for an unrecognized reason — giving up, see $log_file" >&2
      break
    fi
    echo "  WARNING: job $name's nanobot process exited $rc on attempt $((attempt + 1)) for an unrecognized reason — retrying ($((RETRIES - attempt)) retries left)" >&2
    mv "$log_file" "${log_file}.attempt$((attempt + 1))"
    attempt=$((attempt + 1))
  done
  echo "=== [$(date '+%H:%M:%S')] job $name finished -> $log_file ==="
  sync_job_results "$name"
done

echo "All jobs done. Logs in $LOG_DIR"
