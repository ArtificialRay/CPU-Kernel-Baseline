#!/usr/bin/env bash
# Sequentially run one headless Claude Code kernel-optimization session per
# definition, one at a time. Claude Code analogue of bench_nanobot_fleet.sh
# (same JOBS discovery, same best-effort sync-results-after-each-job).
#
# Default: runs every definition found for DATASET. Pass DEFINITIONS to
# scope the run to a fixed allow-list instead (e.g. recovering a specific
# failure set without re-running everything). Accepts a JSON array,
# space-separated bare names, or full "<dataset>_<isa>_<name>" log-file
# stems (the dataset_isa_ prefix is stripped automatically):
#   DEFINITIONS="conv2d_w8a8ch_kh1_kw1_sh1_sw1_dh1_dw1_p0" ./bench_claude_fleet.sh
#
# Prerequisite: an mcp_app session must already be up, launched with
# --author matching AUTHOR below (submit() authorship is fixed server-side
# at server startup, not something this client can override per-request):
#   python3 skills/launch/launch_session.py launch \
#       --isa <isa> --dataset <dataset> --author claude-code --local-port <fixed-port>
#   MCP_ENDPOINT=http://127.0.0.1:<port>/mcp DATASET=<dataset> ISA=<isa> \
#       ./bench_claude_fleet.sh
#
# Headless Claude Code has no hard, code-enforced cap on tool-call rounds —
# MIN_ITERATIONS/MAX_ITERATIONS below are prompt-level only; --max-budget-usd
# is the one real dollar-denominated backstop.
#
# --permission-mode bypassPermissions runs every job unattended, so the
# compile/evaluate tool surface is remote code execution against whatever
# MCP_ENDPOINT points at — only point this at instances/networks you trust.
# --disallowedTools is a denylist, not an allowlist: allowlisting tools for
# a KernelSession silently breaks resource listing/reading
# (list_resources/read_resource) since it's easy to omit a built-in tool
# name you didn't know existed.
#
# Ground rules/workflow live in skills/claude-code/claude-code-kernel-session/
# SKILL.md, read at startup and appended as a system prompt to every job
# (Claude Code has no nanobot-style always-inject mechanism, so this script
# does it explicitly). Keep in sync by hand with nanobot-kernel-session's
# SKILL.md if either changes.
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
# Floor, not a cap on its own — nothing enforces it, the model is just told
# not to submit early. MAX_ITERATIONS = MIN_ITERATIONS+10, a soft ceiling so
# a job doesn't run away once already past the floor. Override MAX_ITERATIONS
# directly if +10 isn't the gap you want.
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
# Defensive backstop for transient infra failures (dropped MCP transport,
# "API returned an empty or malformed response" / StreamIdleTimeoutError).
# Each retry is a brand-new session (--no-session-persistence), but that's
# cheap: prior compile/evaluate results already live server-side in
# bench-trace, and the new session's first move is reading the definition's
# trajectory.jsonl/vN.cpp resources to catch up rather than restarting from
# v1. Total attempts = RETRIES+1. A failed attempt's log is preserved as
# "<log_file>.attemptN" before the next attempt overwrites log_file.
RETRIES="${RETRIES:-3}"

# Must match whatever --author the mcp_app server was launched with. Kept
# distinct from nanobot's default ("nanobot") so the two fleets' solutions/
# traces never collide in bench-trace/.
AUTHOR="${AUTHOR:-claude-code}"
LOCAL_RESULTS_DIR="${LOCAL_RESULTS_DIR:-$REPO_DIR/agent-runs-claude}"
EVAL_CONFIG="$REPO_DIR/eval/eval_config.json"
LABEL="${LABEL:-${DATASET}-${ISA}}"

# DEFINITIONS: optional allow-list scoping the run — see header note. Empty
# (the default) means every definition found for DATASET.
DEFINITIONS="${DEFINITIONS:-}"

# Best-effort: a sync failure never aborts the rest of the batch. (Duplicated
# from bench_nanobot_fleet.sh rather than sourced so each script stays a
# single self-contained file.)
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

# One-off MCP config naming the session "cpu-kernel-baseline".
# --strict-mcp-config makes this the only MCP server visible to the job (the
# project's own .mcp.json/codegraph stays unloaded).
MCP_CONFIG_FILE="$(mktemp -t claude-fleet-mcp-XXXXXX.json)"
trap 'rm -f "$MCP_CONFIG_FILE"' EXIT
python3 -c "
import json, sys
json.dump({'mcpServers': {'cpu-kernel-baseline': {'type': 'http', 'url': sys.argv[1]}}}, open(sys.argv[2], 'w'))
" "$MCP_ENDPOINT" "$MCP_CONFIG_FILE"

# Ground rules + workflow, appended as a system prompt to every job.
# Unprefixed tool names in SKILL_FILE ("compile"/"evaluate"/...) are exposed
# to the model as mcp__cpu-kernel-baseline__*.
SYSTEM_PROMPT="$(cat "$SKILL_FILE")"

# Build JOBS: one "<definition_name>|<prompt>" entry per definition JSON
# under DEFINITIONS_DIR whose baseline-solution dataset (or, for simd-loop
# definitions, whose "simd-loop" tag) matches DATASET, narrowed to
# DEFINITIONS if given. Same discovery logic and baseline_author table as
# bench_nanobot_fleet.sh — kept in sync by hand.
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
  # One JSON event per line as it happens, not buffered until the job
  # finishes. --verbose is required whenever --print is combined with
  # --output-format stream-json.
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
    # tee so stream-json events also print live; PIPESTATUS[0] is claude's
    # exit code, not tee's.
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
