---
name: gen-workload
description: Add new workloads for a definition family; research real inference operating points first, present the plan to the user, then generate with gen_workload.py
---

## gen_workload

Add workloads to definitions in a given family (op type). Before generating anything,
research real inference scenarios for the target op family and present the planned sizes
and their origins to the user for approval.

Script: `scripts/gen_workload.py`

---

## What to extract from the user's message

Skills have no formal parameters — extract context from the user's natural language request:

- **Op family** — which operator type to expand (e.g. `conv2d`, `conv1d`, `deconv2d`,
  `conv2d_depthwise`). If not mentioned, ask before proceeding.
- **Scope** — specific definition names to target, or all definitions in the family (default).

Valid invocations look like:
```
Add more workloads to conv2d
Add real inference sizes for all deconv2d definitions
Add YOLOv5 workloads to conv2d_kh3_kw3_sh1_sw1_dh1_dw1_c64_c128
```

---

## What a workload JSONL looks like

Every definition has a JSONL file at:
```
bench-trace/workloads/<op_family>/<def_name>.jsonl
```
One JSON object per line, one workload per line.

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

Key constraints:
- `axes` covers only the **var** dimensions of the definition. For ncnn op types:
  - conv2d, deconv2d → `N, C_in, H, W`
  - conv1d → `C_in, W` (no N)
  - conv2d_depthwise, deconv2d_depthwise → `N, C, H, W`
  - simd-loop → `N` only
  - Const dimensions (C_out, Kh, Kw, Sh, Sw, Dh, Dw, pad_top, pad_left, pad, activation_type)
    live in the definition JSON — do **NOT** add them to workload `axes` or `inputs`.
- `inputs` must contain **every key** declared in `definition.inputs`:
  - `shape != null` → `{"type": "random"}` — tensor generated at runtime from resolved axes
  - `shape == null` → `{"type": "scalar", "value": v}` — scalar constant passed directly
  - For current ncnn definitions, all inputs are tensors (random). Pad and activation_type
    are const axes — they do not appear in workload inputs at all.
- `uuid` is stable: `uuid5(NAMESPACE_DNS, json.dumps({"def": name, "axes": axes}, sort_keys=True)).hex`
- ncnn adapter only supports **N=1**. Never add N>1 workloads for ncnn-backed definitions.

---

## Workflow

### Step 1: List existing definitions and their current workloads

```bash
ls bench-trace/definitions/<op_family>/
cat bench-trace/workloads/<op_family>/<def_name>.jsonl
```

Identify which definitions exist and how many workloads each already has.

---

### Step 2: Research real inference operating points

For each definition (or group of definitions with the same kernel config), identify
2–3 spatial sizes drawn from real models. Size sources must be concrete, not toy values:

- For conv2d: input feature map `(H, W)` at the layer's position in a named model
  (e.g. ResNet-50 stage2 → H=56, W=56; YOLOv5-s P3 neck → H=80, W=80)
- For conv1d: sequence length `L` from audio/NLP inference (e.g. speech frame = 512,
  phoneme chunk = 128)
- For deconv2d / depthwise: upsampling resolution from segmentation or super-resolution
  models (e.g. DeepLab decoder → H=128, W=128)
- For simd-loop: element count `N` at realistic data sizes (e.g. 4096 = one attention
  head row, 32768 = audio frame)

Skip sizes already covered by existing workloads.

**Required coverage rules — apply across the full set of new workloads:**

1. **Non-divisible channel width** — include at least one workload per definition family
   where `C_in` (or `C`) is not a multiple of common SIMD widths (e.g. `C_in=11` or
   `C_in=3`). These exercise the scalar "remain" tail in hand-written SIMD kernels and
   catch off-by-one bugs in channel-loop cleanup code.
   - Good values: 3 (ResNet/MobileNet first layer), 11 (arbitrary non-power-of-2)
   - The kh7/sh2 definition intentionally uses `C_in=3` (first-layer specialisation path)

2. **Odd spatial sizes — 15–20% of total workloads** — across the full op family,
   15–20% of workload entries must have at least one odd spatial dimension. These stress
   stride=2 boundary logic and deconv output-size edge cases. Good values:
   - H=113, W=113 (prime-ish, common in benchmarks)
   - H=57, W=57 (stride=2 → 29 output, non-power-of-2)
   - H=29, W=29 (deconv boundary)
   Count existing odd-spatial workloads before planning to hit the target without overcounting.

---

### Step 3: Present the plan to the user

Before writing any files, output a table showing what will be generated and why.
Include a coverage summary at the bottom showing non-divisible channel and odd-spatial counts:

```
Planned new workloads for conv2d:

Definition                                        New axes                  Source
conv2d_kh3_kw3_sh1_sw1_dh1_dw1_cout128          N=1, C_in=64,  H=80, W=80   YOLOv5-s P3 neck (640/8)
conv2d_kh3_kw3_sh1_sw1_dh1_dw1_cout128          N=1, C_in=11,  H=56, W=56   non-divisible channel (remain branch)
conv2d_kh3_kw3_sh1_sw1_dh1_dw1_cout128          N=1, C_in=64,  H=113, W=113  odd spatial (stride-2 boundary)
conv2d_kh7_kw7_sh2_sw2_dh1_dw1_cout64           N=1, C_in=3,   H=224, W=224  ResNet first layer (C_in=3 special path)
conv2d_kh7_kw7_sh2_sw2_dh1_dw1_cout64           N=1, C_in=64,  H=57,  W=57   odd spatial (stride-2 boundary)
...

Coverage summary:
  Total new workloads:          12
  Non-divisible C_in (e.g. 11, 3):  2 / 12  (17%)
  Odd spatial (H or W odd):         3 / 12  (25%)  ← target 15–20%

Proceed? (review sizes, sources, and coverage above before confirming)
```

Wait for explicit user confirmation before running the script.

---

### Step 4: Generate

```bash
# Dry-run to confirm output matches the plan
python scripts/gen_workload.py <def_name> \
    --add N=1,C_in=64,H=80,W=80 \
    --add N=1,C_in=11,H=56,W=56 \
    --add N=1,C_in=64,H=113,W=113 \
    --dry-run

# Write for real
python scripts/gen_workload.py <def_name> \
    --add N=1,C_in=64,H=80,W=80 \
    --add N=1,C_in=11,H=56,W=56 \
    --add N=1,C_in=64,H=113,W=113
```

Notes:
- Always include `C_in` (or `C` for depthwise) in every `--add` axis set.
- For conv1d, omit `N` and put `C_in` first: `--add C_in=64,W=512`
- `--scalar` is no longer needed for pad or activation_type — these are const axes
  defined at the definition level, not workload inputs.

Repeat for each definition in the family.

---

### Step 5: Verify

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

## Common bugs

| Symptom | Root cause | Fix |
|---------|------------|-----|
| `ValueError: Workload missing required input var axes: ['C_in']` | `C_in` not included in `--add` axes | Always pass `C_in=...` (or `C=...` for depthwise) in every `--add` |
| `Definition not found: <name>` | Name typo or definition not yet generated | Check `bench-trace/definitions/` — run `gen_definitions.py` if missing |
| `[skip] axes=... already present` | Axes collision with existing workload | Choose different `C_in`/`H`/`W` values |
| `NotImplementedError: ncnn binding supports N=1 only` at eval time | Workload has N>1 for an ncnn-backed definition | Replace with N=1 + different C_in/H/W |
| `KeyError: scalar input 'pad_top' required` | Running `gen_workload.py` against old definitions that had pad as scalar inputs | Regenerate definitions with `gen_definitions.py` — pad is now a const axis |
