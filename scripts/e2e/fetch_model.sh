#!/usr/bin/env bash
# fetch_model.sh <model-key> [models-dir] — download the GGUF for a config/e2e_models.json entry
# (resumable, atomic rename). Run on the measurement box: ~/models/<file>.
set -euo pipefail
KEY="${1:?model key (see config/e2e_models.json)}"; DIR="${2:-$HOME/models}"
HERE="$(cd "$(dirname "$0")" && pwd)"; CFG="$HERE/../../config/e2e_models.json"
read -r REPO FILE < <(python3 -c "import json,sys; m=json.load(open('$CFG'))['$KEY']; print(m['repo'], m['file'])")
mkdir -p "$DIR"; cd "$DIR"
if [ -f "$FILE" ]; then echo "[fetch] $FILE present"; exit 0; fi
echo "[fetch] $REPO/$FILE -> $DIR"
curl -sSL -C - -o "$FILE.part" "https://huggingface.co/$REPO/resolve/main/$FILE"
mv -f "$FILE.part" "$FILE"; ls -la "$FILE"
