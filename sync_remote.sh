#!/usr/bin/env bash
# Push local arm-bench/ + CPU-Kernel-Baseline/ncnn/ to the running c7g instance.
# Mirrors the rsync logic of eval/provision.py (provision + provision_codebase).
#
# Usage:
#   ./sync_remote.sh                # sync both repos
#   ./sync_remote.sh --mirror       # also delete remote files missing locally
#   HOST=1.2.3.4 ./sync_remote.sh   # override host (default: from eval_config.json)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NCNN_ROOT="$REPO_ROOT/../ncnn"
CONFIG="$REPO_ROOT/eval/eval_config.json"
KEY="${KEY:-$HOME/.ssh/id_rsa}"
USER_NAME="${USER_NAME:-ubuntu}"

if [[ -z "${HOST:-}" ]]; then
    if [[ ! -f "$CONFIG" ]]; then
        echo "error: $CONFIG not found and HOST env not set" >&2
        exit 1
    fi
    HOST=$(python3 -c "import json,sys; print(json.load(open('$CONFIG'))['instances']['c7g']['host'])")
fi
if [[ -z "$HOST" ]]; then
    echo "error: empty host (instance not provisioned?)" >&2
    exit 1
fi

EXTRA=()
if [[ "${1:-}" == "--mirror" ]]; then
    EXTRA+=(--delete)
    echo "[sync] mirror mode: remote-only files will be deleted"
fi

SSH_OPTS="ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

echo "[sync] arm-bench/  -> $USER_NAME@$HOST:arm-bench/"
rsync -avz "${EXTRA[@]}" -e "$SSH_OPTS" \
  --exclude=build --exclude=.git --exclude=terraform \
  --exclude=generations --exclude=results --exclude=__pycache__ --exclude='*.pyc' \
  "$REPO_ROOT/" "$USER_NAME@$HOST:arm-bench/"

if [[ -d "$NCNN_ROOT" ]]; then
    echo "[sync] ncnn/      -> $USER_NAME@$HOST:ncnn/"
    rsync -avz "${EXTRA[@]}" -e "$SSH_OPTS" \
      --exclude=build --exclude=.git --exclude=__pycache__ \
      --exclude='*.o' --exclude='*.d' --exclude='*.pyc' \
      "$NCNN_ROOT/" "$USER_NAME@$HOST:ncnn/"
else
    echo "[sync] skipped ncnn (not found at $NCNN_ROOT)"
fi

# Root-level requirements.txt lives at CPU-Kernel-Baseline/requirements.txt
# (not under arm-bench/). Push it to the remote home so `pip3 install -r
# ~/requirements.txt` works after sync.
REQS="$REPO_ROOT/../requirements.txt"
if [[ -f "$REQS" ]]; then
    echo "[sync] requirements.txt -> $USER_NAME@$HOST:~/requirements.txt"
    rsync -az -e "$SSH_OPTS" "$REQS" "$USER_NAME@$HOST:requirements.txt"
fi

echo "[sync] done."