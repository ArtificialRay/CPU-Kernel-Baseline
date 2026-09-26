#!/usr/bin/env bash
# quantref.sh — perplexity across the Qwen3.5-4B quantization ladder on the same text,
# chunk count, threads and binary as the kernel measurement, so "+1.0% perplexity" can be
# read against what a quantization step actually costs. Stock build only, no overrides.
set -u
cd ~/arm-bench-e2e
source ~/miniforge3/etc/profile.d/conda.sh; conda activate armbench-run
LABEL=e2e-quantref; TS=$(date -u +%Y%m%dT%H%MZ); OUT=~/e2e/quantref; mkdir -p $OUT
FILES="Qwen3.5-4B-Q3_K_M.gguf Qwen3.5-4B-Q4_K_S.gguf Qwen3.5-4B-Q4_K_M.gguf Qwen3.5-4B-Q5_K_M.gguf Qwen3.5-4B-Q6_K.gguf Qwen3.5-4B-Q8_0.gguf Qwen3.5-4B-BF16.gguf"
say(){ echo "[qr $(date -u +%H:%M:%S)] $*"; }

say "provisioning c8g.4xlarge (32 GB, needed for the BF16 ceiling)"
python eval/provision.py --isa sve2 --instance c8g.4xlarge --on-demand --label $LABEL --dataset llama.cpp || { say "provision FAILED"; exit 1; }
BOX=$(python -c "
import json, pathlib
for p in ('arm-bench-e2e/eval/eval_config.json','arm-bench/eval/eval_config.json'):
    f = pathlib.Path.home()/p
    if not f.exists(): continue
    h = (json.load(open(f)).get('instances',{}).get('$LABEL') or {}).get('host','')
    if h: print(h); break
")
[ -z "$BOX" ] && { say "no box ip"; exit 1; }
SSHB="ssh -o StrictHostKeyChecking=accept-new ubuntu@$BOX"
say "box $BOX"
$SSHB 'sudo shutdown -c 2>/dev/null; sudo shutdown -h +180' || true
python scripts/e2e/provision_e2e.py --label $LABEL --model qwen3.5-4b || { say "provision_e2e FAILED (box kept)"; exit 1; }

say "shipping scripts and the wikitext file"
$SSHB 'mkdir -p ~/arm-bench/scripts/e2e ~/models'
rsync -az -e "ssh -o StrictHostKeyChecking=accept-new" ~/arm-bench-e2e/scripts/e2e/ ubuntu@$BOX:arm-bench/scripts/e2e/
$SSHB 'test -f ~/models/wiki.test.raw || (cd /tmp && curl -sL -o wt.zip https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip && python3 -c "
import zipfile; zipfile.ZipFile(\"/tmp/wt.zip\").extractall(\"/tmp/wt\")" && cp /tmp/wt/wikitext-2-raw/wiki.test.raw ~/models/wiki.test.raw)'
$SSHB 'df -h / | tail -1'

say "running the ladder (downloads, measures and deletes each in turn)"
$SSHB "cd ~/arm-bench && python3 scripts/e2e/quant_reference.py \
  --bin /home/ubuntu/llama.cpp-e2e/build-stock/bin/llama-perplexity \
  --repo unsloth/Qwen3.5-4B-GGUF --files $FILES \
  --text /home/ubuntu/models/wiki.test.raw --chunks 8 --threads 16 \
  --reference Qwen3.5-4B-Q4_K_M.gguf --out /home/ubuntu/quantref.json" 2>&1 | tee $OUT/table_$TS.txt

scp -q ubuntu@$BOX:~/quantref.json $OUT/quantref_$TS.json || { say "COULD NOT COLLECT -- box $BOX kept"; exit 2; }
say "results -> $OUT/quantref_$TS.json"
say "tearing down"
python eval/provision.py --teardown --label $LABEL || say "TEARDOWN FAILED -- DESTROY $LABEL MANUALLY"
say "done"
