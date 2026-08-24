#!/usr/bin/env bash
# Runs one headless Claude Code kernel-optimization session per definition,
# sequentially. Claude Code analogue of bench_nanobot_fleet.sh.
#
# Prerequisite: an mcp_app session already up, launched with --author
# matching AUTHOR below:
#   python3 skills/launch/launch_session.py launch \
#       --isa <isa> --dataset <dataset> --author claude-code --local-port <fixed-port>
#   MCP_ENDPOINT=http://127.0.0.1:<port>/mcp DATASET=<dataset> ISA=<isa> \
#       ./bench_claude_fleet.sh
#
# --permission-mode bypassPermissions runs every job unattended — only point
# MCP_ENDPOINT at instances/networks you trust (compile/evaluate is remote
# code execution). --disallowedTools is a denylist, not an allowlist —
# allowlisting breaks MCP resource listing.
#
# Ground rules/workflow: skills/claude-code/claude-code-kernel-session/SKILL.md,
# appended as a system prompt to every job. Keep in sync by hand with
# nanobot-kernel-session/SKILL.md.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_DIR/harness_trajs/claude"
DEFINITIONS_DIR="$REPO_DIR/bench-trace/definitions"
SKILL_FILE="$REPO_DIR/skills/claude-code/claude-code-kernel-session/SKILL.md"

if ! command -v claude >/dev/null 2>&1; then
  echo "claude CLI not found on PATH — install Claude Code first." >&2
  exit 1
fi
if [ ! -f "$SKILL_FILE" ]; then
  echo "SKILL_FILE not found: $SKILL_FILE" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Global knobs — override via env, e.g.
#   DATASET=simd-loop MIN_ITERATIONS=60 ISA=sve2 \
#     MCP_ENDPOINT=http://127.0.0.1:9001/mcp ./bench_claude_fleet.sh
# ---------------------------------------------------------------------------
DATASET="${DATASET:-ncnn}"
ISA="${ISA:-sve}"
# Floor, not a cap — model is told not to submit early. MAX_ITERATIONS
# defaults to +10 as a soft ceiling.
MIN_ITERATIONS="${MIN_ITERATIONS:-40}"
MAX_ITERATIONS="${MAX_ITERATIONS:-$((MIN_ITERATIONS + 10))}"
MCP_ENDPOINT="${MCP_ENDPOINT:-}"
if [ -z "$MCP_ENDPOINT" ]; then
  echo "MCP_ENDPOINT is required — launch a session first (skills/launch/launch_session.py" \
       "launch --isa $ISA --dataset $DATASET --author claude-code --local-port <fixed-port>)" \
       "and export the printed endpoint as MCP_ENDPOINT, e.g. http://127.0.0.1:<port>/mcp" >&2
  exit 1
fi
MAX_BUDGET_USD="${MAX_BUDGET_USD:-}"   # optional hard $ ceiling per job
MODEL="${MODEL:-}"                      # optional --model override; empty = CLI default
# Retries transient infra failures (dropped MCP transport, malformed API
# responses). Fresh session each retry — cheap, since compile/evaluate
# results already persist server-side and the new session catches up from
# trajectory.jsonl/vN.cpp. Total attempts = RETRIES+1.
RETRIES="${RETRIES:-3}"

AUTHOR="${AUTHOR:-claude-code}"   # must match the mcp_app server's --author
LOCAL_RESULTS_DIR="${LOCAL_RESULTS_DIR:-$REPO_DIR/agent-runs-claude}"
EVAL_CONFIG="$REPO_DIR/eval/eval_config.json"
LABEL="${LABEL:-${DATASET}-${ISA}}"

# Edit this list to run only specific definitions (paste bare names, or full
# "<dataset>_<isa>_<name>" log-file stems) — leave empty to run every
# definition found for DATASET. Override with the DEFINITIONS env var for a
# one-off run without editing the file.
if [ -z "${DEFINITIONS:-}" ]; then
  DEFINITIONS='
[
]
'
fi

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

# Names the MCP session "cpu-kernel-baseline"; --strict-mcp-config keeps it
# the only MCP server visible to the job.
MCP_CONFIG_FILE="$(mktemp -t claude-fleet-mcp-XXXXXX.json)"
trap 'rm -f "$MCP_CONFIG_FILE"' EXIT
python3 -c "
import json, sys
json.dump({'mcpServers': {'cpu-kernel-baseline': {'type': 'http', 'url': sys.argv[1]}}}, open(sys.argv[2], 'w'))
" "$MCP_ENDPOINT" "$MCP_CONFIG_FILE"

# Ground rules/workflow appended as system prompt; tool names become
# mcp__cpu-kernel-baseline__*.
SYSTEM_PROMPT="$(cat "$SKILL_FILE")"

# One "<definition_name>|<prompt>" per definition JSON matching DATASET
# (narrowed to DEFINITIONS if given). Keep baseline_author in sync by hand
# with bench_nanobot_fleet.sh.
TASK_TEMPLATE='Optimize the "%s" kernel definition (dataset: %s, baseline solution source: %s) in ISA %s. You must spend at least %s tool calls but not exceed %s tool calls to explore genuinely different optimization attempts before you are allowed to submit. once you hit that ceiling, stop iterating and submit your best version immediately, since every iteration spends real model API budget. Follow the ground rules and workflow in your system prompt.'

mapfile -t JOBS < <(python3 - "$DATASET" "$MIN_ITERATIONS" "$DEFINITIONS_DIR" "$TASK_TEMPLATE" "$ISA" "$DEFINITIONS" "$MAX_ITERATIONS" <<'PYEOF'
import json, sys
from pathlib import Path

dataset, min_iterations, definitions_dir, template, isa, definitions_filter, max_iterations = sys.argv[1:8]

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
    prompt = template % (name, ds, baseline_author, isa, min_iterations, max_iterations)
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

mkdir -p "$LOG_DIR" "$LOCAL_RESULTS_DIR"
cd "$REPO_DIR"

echo "Running ${#JOBS[@]} job(s) for DATASET=$DATASET, ISA=$ISA, MIN_ITERATIONS=$MIN_ITERATIONS, MAX_ITERATIONS=$MAX_ITERATIONS (DEFINITIONS=${DEFINITIONS:-<all>})"

CLAUDE_ARGS=(
  -p
  --mcp-config "$MCP_CONFIG_FILE"
  --strict-mcp-config
  --permission-mode bypassPermissions
  --disallowedTools "Bash" "Task" "WebFetch" "WebSearch"
  --append-system-prompt "$SYSTEM_PROMPT"
  --no-session-persistence
  # Streams events live instead of buffering until the job finishes;
  # --verbose is required alongside --output-format stream-json.
  --output-format stream-json
  --verbose
)
[ -n "$MODEL" ] && CLAUDE_ARGS+=(--model "$MODEL")
[ -n "$MAX_BUDGET_USD" ] && CLAUDE_ARGS+=(--max-budget-usd "$MAX_BUDGET_USD")

for job in "${JOBS[@]}"; do
  name="${job%%|*}"
  prompt="${job#*|}"
  log_file="$LOG_DIR/${DATASET}_${ISA}_${name}.log"

  attempt=0
  while true; do
    echo "=== [$(date '+%H:%M:%S')] starting job: $name (attempt $((attempt + 1))/$((RETRIES + 1))) ==="
    set +e
    # tee streams live; PIPESTATUS[0] is claude's exit code, not tee's.
    claude "${CLAUDE_ARGS[@]}" "$prompt" 2>&1 | tee "$log_file"
    rc=${PIPESTATUS[0]}
    set -e
    if [ "$rc" -eq 0 ]; then
      break
    fi
    if [ "$attempt" -ge "$RETRIES" ]; then
      echo "  ERROR: job $name's claude process exited $rc after $((attempt + 1)) attempt(s) — giving up, see $log_file" >&2
      break
    fi
    echo "  WARNING: job $name's claude process exited $rc on attempt $((attempt + 1)) — retrying ($((RETRIES - attempt)) retries left)" >&2
    mv "$log_file" "${log_file}.attempt$((attempt + 1))"
    attempt=$((attempt + 1))
  done
  echo "=== [$(date '+%H:%M:%S')] job $name finished -> $log_file ==="
  sync_job_results "$name"
done

echo "All jobs done. Logs in $LOG_DIR"
