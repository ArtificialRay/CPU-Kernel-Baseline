#!/usr/bin/env bash
# Sequentially run one nanobot kernel-optimization session per definition,
# one at a time. JOBS is generated (not hand-edited) from every definition
# JSON under bench-trace/definitions/ whose baseline-solution dataset
# matches DATASET below (optionally narrowed to DEFINITIONS). Each job's
# stdout/stderr goes straight into its own log file under harness_trajs/nanobot/.
#
# DATASET must be one of the dataset(s) the connected MCP server was started
# with (~/.nanobot/config.json's tools.mcpServers entry runs
# `python3 -m mcp_app.server --dataset <dataset> [--dataset <dataset> ...]`,
# possibly serving several at once via a dispatcher — see
# mcp_app/agent_tools/dispatcher.py) — a job for a definition from a dataset
# the server wasn't started with won't find its resources in that session.
# See skills/nanobot/nanobot-kernel-session/SKILL.md.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_DIR/harness_trajs/nanobot"
NANOBOT_DIR="$HOME/l3/CPU-Kernel-Baseline/nanobot"
DEFINITIONS_DIR="$REPO_DIR/bench-trace/definitions"

# Per-job workspace isolation: each job gets its own nanobot workspace under
# JOB_WORKSPACES_DIR so memory/ and sessions/ never bleed between unrelated
# jobs. Shared/static parts (persona docs + skills) are copied in from
# GLOBAL_WORKSPACE. Job workspaces must live outside any git repo — nanobot's
# GitStore refuses to init a memory-versioning repo when already nested
# inside one (see nanobot/utils/gitstore.py::_is_inside_git_repo).
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
  # Copied (not symlinked) to fit sandboxing requirements.
  for shared in AGENTS.md HEARTBEAT.md SOUL.md USER.md prompts skills; do
    if [ -e "$GLOBAL_WORKSPACE/$shared" ]; then
      rsync -a --delete --exclude='.git' "$GLOBAL_WORKSPACE/$shared" "$job_ws/"
    fi
  done
  echo "$job_ws"
}

# Best-effort: a sync failure never aborts the rest of the batch.
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
# Global knobs — override via env, e.g. `DATASET=simd-loop MIN_ITERATIONS=60 ISA=sve2 ./bench_nanobot_fleet.sh`
#
# DEFINITIONS: optional allow-list to scope the run to specific definitions
# (e.g. recovering a failure set from a prior sweep) instead of the full
# DATASET. Empty (the default) means every definition found for DATASET.
# Accepts a JSON array, bare names or full "<dataset>_<isa>_<name>" log-file
# stems (the dataset_isa_ prefix is stripped automatically), or
# space-separated bare names:
#   DEFINITIONS="gemm_bf16_n1408_k2048 gemm_bf16_n2048_k1024" ./bench_nanobot_fleet.sh
# ---------------------------------------------------------------------------
DATASET="${DATASET:-ncnn}"
# Floor, not a cap — nothing server-side enforces it, the agent is just told
# not to submit early (see PROMPT_TEMPLATE below).
MIN_ITERATIONS="${MIN_ITERATIONS:-40}"
ISA="${ISA:-sve}"
DEFINITIONS="${DEFINITIONS:-}"
# Sandboxed config (restrict_to_workspace + bwrap), kept separate from the
# interactive ~/.nanobot/config.json. Requires `bwrap` installed (sudo
# apt-get install bubblewrap) — otherwise every exec call fails.
NANOBOT_CONFIG="$REPO_DIR/skills/nanobot/nanobot-kernel-session/config.json"
# Defensive backstop against transient infra failures (dropped MCP
# transport, connection errors). Each retry is a brand-new nanobot session,
# but that's cheap: prior compile/evaluate results already live server-side
# in bench-trace, and the new session picks up existing
# vN.cpp/trajectory.jsonl instead of restarting from v1. Total attempts =
# RETRIES+1.
RETRIES="${RETRIES:-3}"

# Sync each job's results back right after it finishes, so already-finished
# jobs survive even if a later job's connection dies mid-batch (e.g. AWS
# reclaiming a spot instance). AUTHOR must match whatever --author the MCP
# server this session connects to was launched with (default "nanobot"
# everywhere in this repo). HOST/USER/KEY_FILE are read from
# eval/eval_config.json, keyed by LABEL — must match whatever --label the
# instance serving this DATASET+ISA was launched/provisioned under (default
# f"{DATASET}-{ISA}"). Override LABEL explicitly if you used a custom
# --label.
AUTHOR="${AUTHOR:-nanobot}"
LOCAL_RESULTS_DIR="${LOCAL_RESULTS_DIR:-$REPO_DIR/agent-runs-$AUTHOR}"
EVAL_CONFIG="$REPO_DIR/eval/eval_config.json"
LABEL="${LABEL:-${DATASET}-${ISA}}"

PROMPT_TEMPLATE='Optimize the "%s" kernel definition (dataset: %s, baseline solution source: %s) in new ISA %s. You must spend at least %s compile+evaluate iterations exploring genuinely different optimization attempts before you are allowed to submit — do not submit early just because an attempt already looks good, keep iterating until you hit the floor. You may keep going past it if you are still finding improvements. Follow the nanobot-kernel-session skill workflow end to end.'

# Build JOBS: one "<definition_name>|<prompt>" entry per definition JSON
# under DEFINITIONS_DIR whose baseline-solution dataset (or, for simd-loop
# definitions, whose "simd-loop" tag) matches DATASET, narrowed to
# DEFINITIONS if given. baseline_author mirrors the dataset/baseline_author
# table hand-maintained in SKILL.md §3.
mapfile -t JOBS < <(python3 - "$DATASET" "$MIN_ITERATIONS" "$DEFINITIONS_DIR" "$PROMPT_TEMPLATE" "$ISA" "$DEFINITIONS" <<'PYEOF'
import json, sys
from pathlib import Path

dataset, min_iterations, definitions_dir, template, isa, definitions_filter = sys.argv[1:7]

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
    prompt = template % (name, ds, baseline_author, isa, min_iterations)
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

echo "Running ${#JOBS[@]} job(s) for DATASET=$DATASET, ISA=$ISA, MIN_ITERATIONS=$MIN_ITERATIONS (DEFINITIONS=${DEFINITIONS:-<all>})"

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
    # Known benign case: nanobot's close_mcp() can crash on
    # asyncio.CancelledError while tearing down two concurrent MCP sessions
    # (only when two datasets' fleets run in parallel against two live
    # instances) — happens AFTER the job's own work is done and
    # evaluate() has already auto-persisted the best version remotely, so
    # treat as a warning and move on rather than letting `set -e` abort the
    # whole batch.
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
