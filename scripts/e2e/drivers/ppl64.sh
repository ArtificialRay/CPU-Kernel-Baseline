#!/usr/bin/env bash
# ppl64.sh — the headline quality numbers at 64 chunks instead of 8, so the error bars shrink
# ~3x and the quantization ladder can actually be quoted. Same text, threads and binary family
# as every earlier measurement; only --chunks changes.
#   plain models (stock build):      Q3_K_M, Q4_K_M, Q6_K, BF16
#   gate-v2 kernels on Q4_K_M:       build-agent + the manifest built from the agent run dirs
set -u
cd ~/arm-bench-e2e
source ~/miniforge3/etc/profile.d/conda.sh; conda activate armbench-run
LABEL=e2e-ppl64; TS=$(date -u +%Y%m%dT%H%MZ); OUT=~/e2e/ppl64; mkdir -p $OUT
E=~/e2e; MAN=$E/gatev2b/manifest_gatev2b.json
CHUNKS=64; THREADS=16
say(){ echo "[p64 $(date -u +%H:%M:%S)] $*"; }

[ -f "$MAN" ] || { say "no manifest at $MAN -- run measure4 first"; exit 1; }
say "verifying the manifest still holds the re-optimized kernels"
python scripts/e2e/verify_kernel_set.py --manifest $MAN --reference ~/runs-fable-orig \
  --expect-count 11 \
  --expect-changed gemm_ggml_q4_K_n4096_k2560 gemm_ggml_q4_K_n8192_k2560 gemm_ggml_q4_K_n9216_k2560 gemm_ggml_q5_K_n2560_k4096 gemm_ggml_q5_K_n8192_k2560 \
  --expect-same gemm_ggml_q4_K_n1024_k2560 gemm_ggml_q4_K_n2560_k4096 gemm_ggml_q4_K_n2560_k9216 gemm_ggml_q6_K_n1024_k2560 gemm_ggml_q6_K_n248320_k2560 gemm_ggml_q6_K_n2560_k9216 \
  --forbid 'maxabs >> sh' > $OUT/verify_$TS.log 2>&1 || { say "KERNEL SET VERIFICATION FAILED -- nothing provisioned"; tail -8 $OUT/verify_$TS.log; exit 1; }
tail -1 $OUT/verify_$TS.log

say "provisioning c8g.4xlarge"
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
$SSHB 'sudo shutdown -c 2>/dev/null; sudo shutdown -h +300' || true
python scripts/e2e/provision_e2e.py --label $LABEL --model qwen3.5-4b --agent-build || { say "provision_e2e FAILED (box kept)"; exit 1; }

say "shipping kernels, scripts and wikitext"
$SSHB 'mkdir -p ~/e2e/gatev2b ~/arm-bench/scripts/e2e'
rsync -az -e "ssh -o StrictHostKeyChecking=accept-new" --include '*.so' --include 'manifest_gatev2b.json' --exclude '*' $E/gatev2b/ ubuntu@$BOX:e2e/gatev2b/
rsync -az -e "ssh -o StrictHostKeyChecking=accept-new" ~/arm-bench-e2e/scripts/e2e/ ubuntu@$BOX:arm-bench/scripts/e2e/
$SSHB 'test -f ~/models/wiki.test.raw || (cd /tmp && curl -sL -o wt.zip https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip && python3 -c "
import zipfile; zipfile.ZipFile(\"/tmp/wt.zip\").extractall(\"/tmp/wt\")" && cp /tmp/wt/wikitext-2-raw/wiki.test.raw ~/models/wiki.test.raw)'

say "hook check -- a silently disabled override would look exactly like the stock build"
$SSHB "ARMBENCH_OVERRIDES=/home/ubuntu/e2e/gatev2b/manifest_gatev2b.json ARMBENCH_OVERRIDE_LOG=1 ~/llama.cpp-e2e/build-agent/bin/llama-bench -m ~/models/Qwen3.5-4B-Q4_K_M.gguf -p 16 -n 0 -t 4 -r 1" > $OUT/hook_$TS.log 2>&1
h=$(grep -c 'hit mul_mat' $OUT/hook_$TS.log)
say "hook check: $h shapes intercepted"
[ "$h" -lt 11 ] && { say "HOOK DID NOT FIRE ($h/11) -- box $BOX kept"; exit 1; }

say "gate-v2 kernels on Q4_K_M, $CHUNKS chunks"
$SSHB "ARMBENCH_OVERRIDES=/home/ubuntu/e2e/gatev2b/manifest_gatev2b.json ~/llama.cpp-e2e/build-agent/bin/llama-perplexity \
  -m ~/models/Qwen3.5-4B-Q4_K_M.gguf -f ~/models/wiki.test.raw --chunks $CHUNKS -t $THREADS" > $OUT/agent_$TS.log 2>&1
grep -a 'Final estimate' $OUT/agent_$TS.log | tee $OUT/agent_ppl_$TS.txt

say "plain quantization ladder, $CHUNKS chunks"
$SSHB "cd ~/arm-bench && python3 scripts/e2e/quant_reference.py \
  --bin /home/ubuntu/llama.cpp-e2e/build-stock/bin/llama-perplexity \
  --repo unsloth/Qwen3.5-4B-GGUF \
  --files Qwen3.5-4B-Q3_K_M.gguf Qwen3.5-4B-Q4_K_M.gguf Qwen3.5-4B-Q6_K.gguf Qwen3.5-4B-BF16.gguf \
  --text /home/ubuntu/models/wiki.test.raw --chunks $CHUNKS --threads $THREADS --keep \
  --reference Qwen3.5-4B-Q4_K_M.gguf --out /home/ubuntu/ppl64.json" 2>&1 | tee $OUT/table_$TS.txt

scp -q ubuntu@$BOX:~/ppl64.json $OUT/ppl64_$TS.json || { say "COULD NOT COLLECT -- box $BOX kept"; exit 2; }
say "results -> $OUT/ppl64_$TS.json  (agent arm in $OUT/agent_ppl_$TS.txt)"
say "tearing down"
python eval/provision.py --teardown --label $LABEL || say "TEARDOWN FAILED -- DESTROY $LABEL MANUALLY"
say "done"
