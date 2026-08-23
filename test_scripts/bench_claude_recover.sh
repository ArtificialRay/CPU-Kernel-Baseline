#!/usr/bin/env bash
# Re-run one headless Claude Code kernel-optimization session per definition
# for a fixed allow-list of definitions that failed in a prior
# bench_claude_fleet.sh sweep. This is the Claude Code analogue of
# bench_nanobot_recover.sh in this same directory — same DEFINITIONS
# allow-list convention (JSON array or space-separated, dataset_isa_ log-stem
# prefix stripped automatically), same MIN_ITERATIONS-floor +
# MAX_ITERATIONS=MIN_ITERATIONS+10-ceiling prompt framing. Read
# bench_claude_fleet.sh's header first if you haven't; this one only calls
# out where recovery differs from a full fleet sweep.
#
# DEFINITIONS below is the actual failure set diagnosed from
# harness_trajs/claude/*.log for DATASET=ncnn ISA=sve:
#   - MCP server never connected (init message's mcp_servers[0].status was
#     "failed", so mcp__cpu-kernel-baseline__{compile,evaluate,disassemble}
#     never appeared in the job's tool list at all — every subsequent
#     compile attempt errored "No such tool available"): all 6 depthwise
#     fp32/w8a8ch kh3/kh5 defs that had already run, plus kh7_kw7 (fp32 and
#     w8a8ch) and the four remaining w8a8ch kh1_kw1/kh3_kw3 defs.
#   - Never attempted yet (no log file under harness_trajs/claude/ at all):
#     the two depthwise_w8a8ch_kh5_kw5 defs.
#   - MCP connected fine but the session was killed mid-run by the
#     "API returned an empty or malformed response" / StreamIdleTimeoutError
#     transport fault, leaving only 1-2 versions saved: kh1_kw1_sh2_sw2 and
#     kh3_kw3_sh1_sw1 (both fp32).
# Edit the list below (or override DEFINITIONS, see below) if you re-run
# bench_claude_fleet.sh and want to recover a different failure set.
#
# Prerequisite: same as bench_claude_fleet.sh — an mcp_app session must
# already be up, reachable at a known local URL, launched with --author
# claude-code:
#
#   python3 skills/launch/launch_session.py launch \
#       --isa <isa> --dataset <dataset> --author claude-code \
#       --local-port <fixed-port>
#
#   MCP_ENDPOINT=http://127.0.0.1:<port>/mcp DATASET=<dataset> ISA=<isa> \
#       ./bench_claude_recover.sh
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
#   DATASET=ncnn MIN_ITERATIONS=60 ISA=sve \
#     MCP_ENDPOINT=http://127.0.0.1:9001/mcp ./bench_claude_recover.sh
# ---------------------------------------------------------------------------
DATASET="${DATASET:-ncnn}"
ISA="${ISA:-sve}"
# Floor, not a cap on its own — same MIN_ITERATIONS/MAX_ITERATIONS pairing
# as bench_claude_fleet.sh (MAX_ITERATIONS derived as MIN_ITERATIONS+10, a
# soft ceiling so a recovery job doesn't run away and rack up unbounded API
# spend). Soft limit either way: nothing server-side or CLI-side enforces
# it, the model is just told to stay within [floor, ceiling]. Override
# MAX_ITERATIONS directly if +10 isn't the gap you want.
MIN_ITERATIONS="${MIN_ITERATIONS:-40}"
MAX_ITERATIONS="${MAX_ITERATIONS:-$((MIN_ITERATIONS + 10))}"
MCP_ENDPOINT="${MCP_ENDPOINT:-}"
if [ -z "$MCP_ENDPOINT" ]; then
  echo "MCP_ENDPOINT is required — launch a session first (skills/launch/launch_session.py" \
       "launch --isa $ISA --dataset $DATASET --author claude-code --local-port <fixed-port>)" \
       "and export the printed endpoint as MCP_ENDPOINT, e.g. http://127.0.0.1:<port>/mcp" >&2
  exit 1
fi
MAX_BUDGET_USD="${MAX_BUDGET_USD:-}"   # optional hard $ ceiling per job — see fleet script's header note
MODEL="${MODEL:-}"                      # optional --model override; empty = CLI default
# Defensive backstop, not a fix for a known-live bug — see
# bench_claude_fleet.sh's RETRIES comment for the full rationale. Each retry
# is a brand-new session (no --resume), but that's cheap: prior
# compile/evaluate results already live server-side in bench-trace, and the
# new session's first move is reading the definition's
# trajectory.jsonl/vN.cpp resources to catch up — it doesn't restart the
# optimization from v1. Total attempts = RETRIES+1.
RETRIES="${RETRIES:-3}"

AUTHOR="${AUTHOR:-claude-code}"
LOCAL_RESULTS_DIR="${LOCAL_RESULTS_DIR:-$REPO_DIR/agent-runs-claude}"
EVAL_CONFIG="$REPO_DIR/eval/eval_config.json"
LABEL="${LABEL:-${DATASET}-${ISA}}"

# ---------------------------------------------------------------------------
# DEFINITIONS: allow-list of definitions to recover. Override to recover a
# different set (JSON array, bare names or full "<dataset>_<isa>_<name>"
# log-file stems — the dataset_isa_ prefix is stripped automatically — or
# space-separated bare names):
#   DEFINITIONS="conv2d_w8a8ch_kh1_kw1_sh1_sw1_dh1_dw1_p0" ./bench_claude_recover.sh
# ---------------------------------------------------------------------------
if [ -z "${DEFINITIONS:-}" ]; then
  DEFINITIONS='
[
"conv2d_fp32_kh3_kw3_sh1_sw1_dh1_dw1_p1",
"conv2d_fp32_kh3_kw3_sh2_sw2_dh1_dw1_p1",
"conv2d_fp32_kh7_kw7_sh2_sw2_dh1_dw1_p3"
]
'
fi

# Best-effort: a sync failure (or an unreachable/reclaimed instance) is
# logged and swallowed, never fatal — losing the ability to sync one job's
# results shouldn't abort the rest of the batch. (Duplicated from
# bench_claude_fleet.sh rather than sourced so each script stays a single
# self-contained file.)
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
# One-off MCP config file naming the mcp_app session as "cpu-kernel-baseline".
# --strict-mcp-config (used below) makes this the *only* MCP server visible
# to the job — the project's own .mcp.json (codegraph) is intentionally not
# loaded, keeping each job's tool surface scoped to exactly this session.
# ---------------------------------------------------------------------------
MCP_CONFIG_FILE="$(mktemp -t claude-recover-mcp-XXXXXX.json)"
trap 'rm -f "$MCP_CONFIG_FILE"' EXIT
python3 -c "
import json, sys
json.dump({'mcpServers': {'cpu-kernel-baseline': {'type': 'http', 'url': sys.argv[1]}}}, open(sys.argv[2], 'w'))
" "$MCP_ENDPOINT" "$MCP_CONFIG_FILE"

# ---------------------------------------------------------------------------
# Ground rules + workflow, read from SKILL_FILE and appended as a system
# prompt to every job — same as bench_claude_fleet.sh. Tool names inside that
# file are unprefixed ("compile"/"evaluate"/...) — Claude Code exposes them
# to the model as mcp__cpu-kernel-baseline__*, which it resolves itself from
# the connected server's tool schema.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT="$(cat "$SKILL_FILE")"

# ---------------------------------------------------------------------------
# Build JOBS: one "<definition_name>|<prompt>" entry per definition JSON
# under DEFINITIONS_DIR whose baseline-solution dataset (or, for simd-loop
# definitions, whose "simd-loop" tag) matches DATASET, narrowed to
# DEFINITIONS. Same discovery logic and baseline_author table as
# bench_claude_fleet.sh / bench_nanobot_recover.sh — kept in sync by hand.
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE='Optimize the "%s" kernel definition (dataset: %s, baseline solution source: %s) in ISA %s. You must spend at least %s tool calls but not exceed %s tool calls to explore genuinely different optimization attempts before you are allowed to submit. once you hit that ceiling, stop iterating and submit your best version immediately, since every iteration spends real model API budget. Follow the ground rules and workflow in your system prompt.'

mapfile -t JOBS < <(python3 - "$DATASET" "$MIN_ITERATIONS" "$DEFINITIONS_DIR" "$PROMPT_TEMPLATE" "$ISA" "$DEFINITIONS" "$MAX_ITERATIONS" <<'PYEOF'
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

echo "Recovering ${#JOBS[@]} job(s) for DATASET=$DATASET, ISA=$ISA, MIN_ITERATIONS=$MIN_ITERATIONS, MAX_ITERATIONS=$MAX_ITERATIONS"

CLAUDE_ARGS=(
  -p
  --mcp-config "$MCP_CONFIG_FILE"
  --strict-mcp-config
  --permission-mode bypassPermissions
  --disallowedTools "Bash" "Task" "WebFetch" "WebSearch"
  --append-system-prompt "$SYSTEM_PROMPT"
  --no-session-persistence
  # stream-json writes one JSON event per line as it happens instead of
  # buffering until the whole run finishes — see bench_claude_fleet.sh's
  # header note. --verbose is required by the CLI whenever --print is
  # combined with --output-format stream-json.
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
    echo "=== [$(date '+%H:%M:%S')] starting recovery job: $name (attempt $((attempt + 1))/$((RETRIES + 1))) ==="
    set +e
    # tee (not a plain redirect) so stream-json events also print live to the
    # terminal as the job runs, not just after it finishes; PIPESTATUS[0] is
    # needed because $? after a pipeline reflects tee's exit code, not claude's.
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

echo "All recovery jobs done. Logs in $LOG_DIR"
