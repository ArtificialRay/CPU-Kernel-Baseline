#!/usr/bin/env bash
# Save an OpenAI API key to ~/.openai_key without it ever appearing on screen,
# in shell history, or in a chat transcript. Run in your own terminal:
#     bash ~/CMU/ckb-sweep/analysis/save_openai_key.sh
set -euo pipefail
dest="$HOME/.openai_key"
printf 'Paste the OpenAI key and press Enter (input is hidden): '
IFS= read -rs key
printf '\n'
key="${key//[[:space:]]/}"          # strip stray whitespace/newlines
if [[ -z "$key" ]]; then echo "nothing entered, aborting"; exit 1; fi
if [[ "$key" != sk-* ]]; then echo "warning: key does not start with 'sk-' (saved anyway)"; fi
umask 077
printf '%s' "$key" > "$dest"
chmod 600 "$dest"
echo "saved: $dest  (${#key} chars, starts with ${key:0:3}..., perms $(stat -f '%Sp' "$dest"))"
unset key
