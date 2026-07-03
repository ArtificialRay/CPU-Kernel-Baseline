---
name: gen-workload
description: Add new workloads for a definition family; choose the right collection path, run it, then verify
---

## gen_workload

Add workloads to definitions in a given family (op type). The primary path is always
automated collection via a collect script; manual `gen_workload.py --add` calls are
for targeted gap-fills only.

---

## Workflow

### Step 1 — Identify op family and collection path

Ask the user which op_type + definition to target if not stated. Both collector
scripts take `--op-type <op_type> --definition <name>` (both required) and route
by model kernel category:

| Model kernel category | Collection script | Source |
|-----------|------------------|--------|
| CV model kernel | `scripts/gen-workload/collect_workloads_conv.py --op-type <op_type> --definition <name>` | torchvision ResNet50 + MobileNetV3 hooks |
| LLM model kernel | `scripts/gen-workload/collect_workloads_llm.py --model <gguf> --op-type <op_type> --definition <name>` | llama.cpp on remote Graviton + ShareGPT |
| Neither (no torchvision/ggml hook exists for this op_type) | manual `scripts/gen-workload/gen_workload.py --add` (Step 4) | — |

Each script only knows how to capture shapes for a small set of registered op_types.
Run `python scripts/gen-workload/collect_workloads_conv.py --list-op-types` /
`python scripts/gen-workload/collect_workloads_llm.py --list-op-types` to see what's currently supported instead of relying on a hardcoded
list here — passing an unregistered `--op-type` errors out immediately with that same
list. A brand-new op_type under an *already-supported* category (e.g. another `gemm`
or `conv2d` definition) needs no script changes — only a genuinely new capture
mechanism (a new nn.Module type, a new ggml op) requires adding a registry entry.

If the user asks about a single specific definition rather than a whole family, go
directly to Step 4 (manual targeted addition).

---

### Step 2a — Automated collection: CV model kernels (conv2d, conv2d_depthwise, gemm, pooling)

```bash
# Dry-run preview first
python scripts/gen-workload/collect_workloads_conv.py --op-type conv2d \
    --definition conv2d_fp32_kh1_kw1_sh1_sw1_dh1_dw1_p0 --dry-run

# Write workloads
python scripts/gen-workload/collect_workloads_conv.py --op-type conv2d \
    --definition conv2d_fp32_kh1_kw1_sh1_sw1_dh1_dw1_p0
```

The script runs ResNet50 and MobileNetV3-Large across 11 standard CV input resolutions
(96→1024, plus non-square 360×640 and 480×640) and hooks whichever `nn.Module` type the
requested `--op-type` is registered against (`nn.Conv2d` for conv2d/conv2d_depthwise,
`nn.Linear` for gemm, `nn.AdaptiveAvgPool2d`/`nn.MaxPool2d` for pooling). Captured shapes
are matched to the one requested definition by its own const axes (read straight from
the definition JSON — Kh/Kw/Sh/Sw/Dh/Dw/pad_top for conv-shaped ops, K/N for gemm, none
for global-avg pooling), then routed to `gen_workload.py` with `--max-count 20
--batch-key <axis>` (batch-key is chosen automatically per op_type: `C_in` for conv2d,
`C` for conv2d_depthwise/pooling, `M` for gemm). The two-layer filter inside
gen_workload.py (see Step 3) selects diverse operating points automatically.

If a definition's exact const-axis combination doesn't occur in either reference model
(e.g. an odd pooling kernel/stride that ResNet50/MobileNetV3 never use), the script
prints `[skip] ... no matching captures` — fall back to Step 4 manual addition for that
one definition.

`w8a8ch` definitions (any op_type) automatically receive a random-but-bounded
`input_scale` scalar — see the note at the end of Step 2c.

---

### Step 2b — (superseded)

CV gemm (`nn.Linear`) and pooling (`nn.AdaptiveAvgPool2d`/`nn.MaxPool2d`) hooks are now
built into `collect_workloads_conv.py`'s registry — just pass `--op-type gemm` or
`--op-type pooling` as in Step 2a. No manual script editing needed.

---

### Step 2c — Automated collection: LLM model kernels via llama.cpp

```bash
# Requires: Graviton4 instance in eval_config.json + GGUF model on the remote
python scripts/gen-workload/collect_workloads_llm.py \
    --model '~/models/OLMoE-1B-7B-0924-Instruct-Q8_0.gguf' \
    --op-type gemm --definition gemm_fp32_n2048_k2048 \
    --num-prompts 200
```

The script provisions (or reuses) the c8g instance, rsyncs llama.cpp from local
`../llama.cpp/`, builds `collect-ggml-shapes` on the remote, runs it against the GGUF
model with 200 ShareGPT prompts, downloads the shapes JSON, then calls gen_workload.py
for the one requested `--op-type`/`--definition` (registered kinds: `gemm`, `rms_norm`,
`mha`, `moe`). The `--max-count 20 --batch-key M` (or `n_tokens` for moe) filter is
applied automatically.

Notes:
- `n_tokens` for moe = total token batch entering the MoE layer (same range as prompt
  lengths, 1–512); per-expert routing is internal to the kernel.
- `w8a8ch` definitions (any op_type, both scripts) automatically receive a
  `--scalar input_scale=<value>` where `<value>` is drawn from a bounded range
  (`uniform(0.005, 0.05)`) seeded off the definition name — reproducible across
  reruns, not a shared magic constant. `q8_0` definitions need no scalar at all
  (their dequant scales are per-block tensor inputs, not workload-level scalars).

---

### Step 3 — Two-layer workload filter (inside gen_workload.py)

All automated collection calls gen_workload.py with `--max-count N --batch-key <axis>`,
triggering the built-in two-layer filter:

**Layer 1 — dedup** (`--max-dups`, default 2):
Keep at most 2 copies of any identical axes combo, removing the long tail of repeated
shapes that appear across many resolutions.

**Layer 2 — greedy farthest-point diversity** (`--max-count`):
Phase 1: seed one representative per distinct `batch-key` value (ensures every channel
width or M range is covered). Phase 2: greedy fill remaining slots by maximising
normalised Euclidean distance to already-selected points.

When doing manual `--add` calls for small targeted additions (Step 4), **omit
`--max-count`** so all specified axes are written as-is. Use `--max-count` only when
handing a large candidate pool (hundreds of captured shapes) to the script.

---

### Step 4 — Manual targeted additions (for gaps)

Use this path when the automated collector doesn't cover the op family (conv1d,
simd-loop) or a specific operating point is missing.

Present a table of planned workloads and their source/rationale, then wait for explicit
user confirmation before writing.

**Conv2d / depthwise:**
```bash
python scripts/gen-workload/gen_workload.py <def_name> \
    --add N=1,C_in=64,H=80,W=80 \
    --add N=1,C_in=11,H=56,W=56 \
    --dry-run    # preview; remove --dry-run to write
```

Coverage to check before adding:
- At least one `C_in` (or `C`) not a multiple of common SIMD widths (3, 11, etc.)
- 15–20% of total workloads should have an odd spatial dimension (H=57, W=113, etc.)

**LLM gemm / rms_norm:**
```bash
python scripts/gen-workload/gen_workload.py gemm_fp32_n2048_k2048 \
    --add M=1 --add M=64 --add M=256 --add M=512
```

**LLM mha** — always include the decode point M=1, S=max_ctx:
```bash
python scripts/gen-workload/gen_workload.py mha_fp32_h16_d128_kvh16 \
    --add M=1,S=1 --add M=64,S=64 --add M=256,S=256 --add M=1,S=4096
```

**LLM moe:**
```bash
python scripts/gen-workload/gen_workload.py moe_fp32_e64_k8_d2048_ff1024 \
    --add n_tokens=1 --add n_tokens=64 --add n_tokens=256
```

**w8a8ch gemm** — needs scalar not present in existing workloads (0.01 here is just an
illustrative manual value — the automated path in Step 2a/2c draws a bounded random
value instead, see there):
```bash
python scripts/gen-workload/gen_workload.py gemm_w8a8ch_n2048_k2048 \
    --add M=1 --add M=64 --add M=256 \
    --scalar input_scale=0.01
```

**conv1d** — no N axis:
```bash
python scripts/gen-workload/gen_workload.py conv1d_kw3_sh1_dh1_p1_c64_c128 \
    --add C_in=64,W=512 --add C_in=64,W=128
```

---

### Step 5 — Verify

```bash
wc -l bench-trace/workloads/<op_family>/*.jsonl
```

Or via Python:
```bash
python -c "
from bench.data import TraceSet
ts = TraceSet.from_path('bench-trace')
for name in ['<def_name_1>', '<def_name_2>']:
    wls = ts.get_workloads(name)
    print(len(wls), 'workloads OK:', name)
"
```

---

## Supplementary reference

### Workload JSONL format

```json
{
  "axes":   {"N": 1, "C_in": 64, "H": 56, "W": 56},
  "inputs": {
    "input":  {"type": "random"},
    "weight": {"type": "random"}
  },
  "uuid": "1c29650b35bb4caca50fd7acb24b07be",
  "tags": {"from": "gen_workload"}
}
```

`axes` covers only the **var** dimensions from the definition. Const axes (Kh, Kw, K,
N, etc.) live in the definition JSON — do not repeat them here.

Input types: `shape != null` → `{"type": "random"}`; `shape == null` → `{"type":
"scalar", "value": v}`. UUID: `uuid5(NAMESPACE_DNS, json.dumps({"def": name, "axes":
axes}, sort_keys=True)).hex`.

### Var axes by op family

| Op family | Var axes | Notes |
|-----------|----------|-------|
| `conv2d` | `N, C_in, H, W` | N=1 only (ncnn adapter) |
| `conv2d_depthwise` | `N, C, H, W` | C_in = C_out = C; N=1 only |
| `conv1d` | `C_in, W` | no N axis |
| `pooling` | `N, C, H, W` | all four vary with input resolution |
| `simd-loop` | `N` | element count |
| `gemm` | `M` | K, N are const (encoded in def name) |
| `rms_norm` | `M` | D const |
| `mha` | `M, S` | M = query len, S = KV len |
| `moe` | `n_tokens` | total tokens entering MoE layer |

### Common bugs

| Symptom | Root cause | Fix |
|---------|------------|-----|
| `ValueError: Workload missing required input var axes: ['C_in']` | `C_in` omitted from `--add` | Always pass `C_in=...` (or `C=...` for depthwise) |
| `Definition not found: <name>` | Name typo or def not generated yet | Check `bench-trace/definitions/` |
| `[skip] axes=... already present` | Duplicate axes | Choose different values |
| `NotImplementedError: ncnn binding supports N=1 only` | Workload has N>1 for ncnn-backed def | Use N=1 |
| `KeyError: scalar input 'input_scale' required` | `*_w8a8ch_*` has no existing workload to inherit scalar | Add `--scalar input_scale=<value>` (both collector scripts do this automatically) |
| `WARNING: no exps records found` in collect_workloads_llm.py | OLMoE expert ops use `GGML_OP_MUL_MAT_ID` | Expected — not a bug |
| `error: --op-type and --definition are required` | Both collector scripts now require explicit `--op-type`/`--definition` (no more "run everything") | Pass both, or `--list-op-types` to see what's registered |
| `error: Unsupported op_type '<x>' for this collector` | `<x>` has no capture-spec/axes-builder registered in this script | Try the other script (CV vs LLM), or fall back to manual `gen_workload.py --add` (Step 4) |
