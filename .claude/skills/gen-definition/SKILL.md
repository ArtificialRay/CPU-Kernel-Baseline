---
name: gen-definition
description: Define a brand-new op type as a bench-trace Definition JSON; use when the user wants to benchmark an operator not yet in bench-trace/definitions/
---

## gen_definition

Interactively design and write a Definition JSON for any new op type, following the
schema used across all existing definitions in `bench-trace/definitions/`.

**This skill writes the structural schema only.** Workloads (concrete tensor sizes) are
added afterward via `/gen-workload`. The reference function inside the definition JSON
is a correctness oracle — it is not a workload.

---

## What to extract from the user's message

- **Op name** — the user's name for this operation (e.g. `matmul`, `attention_qkv`,
  `rms_norm`). If ambiguous, ask what mathematical operation it performs.
- **Op description** — what the op computes (enough to write a reference function).
- **Axes** — all symbolic dimensions the op uses: which are fixed per kernel config
  (`const`) and which vary per workload (`var`)?
- **Input tensors** — names, shapes (as lists of axis names), and dtypes.
- **Output tensors** — names, shapes, and dtypes.
- **Constraints** — equations that express derived output shape axes in terms of input
  axes and const axes.
- **Reference** — a Python function using NumPy or PyTorch that computes the correct
  output for any valid inputs.

If the user's description is underspecified in any of these areas, ask before generating.

---

## Definition JSON schema

Every definition JSON follows this structure exactly. Study the existing files in
`bench-trace/definitions/` before writing — they are the ground truth.

```json
{
  "name":        "<name>",
  "op_type":     "<op_type>",
  "description": "<human-readable description>",
  "tags":        ["status:active", "baseline-solution:<ncnn|llama.cpp>"],

  "axes": {
    "<axis_name>": { "type": "var" },
    "<axis_name>": { "type": "var", "parent": "<other_var_axis>" },
    "<axis_name>": { "type": "const", "value": <integer> }
  },

  "inputs": {
    "<input_name>": { "shape": ["<axis>", ...], "dtype": "<dtype>" },
    "<scalar_name>": { "shape": null, "dtype": "<dtype>" }
  },

  "outputs": {
    "<output_name>": { "shape": ["<axis>", ...], "dtype": "<dtype>" }
  },

  "constraints": [
    "<derived_axis> == <expression using other axes>"
  ],

  "reference": "<Python source string — a def run(...): function>"
}
```

### Key design rules

**`axes` — var vs. const**

- `"var"` — the value changes between workloads (batch size, spatial dims, channel count
  when it is the sweep axis, sequence length, etc.).
- `"const"` — the value is fixed for all workloads of this definition (kernel size,
  stride, dilation, number of output channels when set at kernel config time, etc.).
- Only `var` axes appear in workload `.axes` dicts; `const` axes do NOT.
- Use `"parent"` on a `var` axis when its value is determined by another `var` axis in
  the same workload row (e.g. `H_out` is always derived from `H`, so it carries
  `"parent": "N"` in the conv2d family, grouping them under the same workload row).

**`inputs` shapes**

- `"shape": ["A", "B", "C"]` — tensor whose sizes come from the listed axes.
  Axis names must all appear in `"axes"`. Both `var` and `const` axes are allowed.
- `"shape": null` — scalar constant (passed as a fixed value in each workload).

**`constraints`**

- One string per constraint. Only needed for axes whose value is **derived** (i.e. a
  `var` axis whose value is fully determined by other axes + const axes).
- Use Python integer floor-division `//` for conv output sizes.
- Axes referenced must all be in `"axes"`.

**`reference`**

- A complete Python source string containing exactly one function named `run`.
- The signature must match the definition's `inputs` — one positional arg per input, in
  declaration order.
- Array inputs arrive as NumPy arrays. Scalar inputs (`shape: null`) arrive as Python ints.
- The return value must match the single output tensor (NumPy array).
- Write correct, minimal PyTorch or NumPy code — no comments, no prints.
- All numeric constants baked from `const` axes must be literal integers (not references
  to axis names).

---

## Naming and output path

- **`name`** — descriptive, lowercase, underscores; encode the key const axes that
  distinguish variants (e.g. `matmul_m512_k256`, `rms_norm_d768`). If the op has no
  const axes that vary between definitions, a plain name is fine (`layer_norm`).
- **`op_type`** — typically the same as the first component of `name` (the op family).
- **Output path** — `bench-trace/definitions/<op_type>/<name>.json`.
  - Use a new subdirectory named after `op_type` if none exists yet.

---

## Workflow

### Step 0: Determine shape source

When the op comes from a real model, extract shapes using one of three methods before
designing the definition. Record the source as a `model:` tag.

**Method A — HuggingFace config.json** (LLM/MoE/transformer models)
```python
from huggingface_hub import hf_hub_download
import json, pathlib
cfg = json.loads(pathlib.Path(hf_hub_download("org/model", "config.json")).read_text())
# hidden_size, num_attention_heads, num_key_value_heads,
# moe_intermediate_size / intermediate_size, num_experts, num_experts_per_tok
```
Downloads config only — no weights. See `scripts/gen-definition/model_shapes.py::extract_hf_shapes`.

**Method B — torchvision forward hooks** (CV models)
```python
import torch, torch.nn as nn, torchvision
model = torchvision.models.resnet50(weights=None).eval()
# register_forward_hook on nn.Linear / nn.AdaptiveAvgPool2d
# run dummy forward, collect shapes
```
No weights downloaded. See `scripts/gen-definition/model_shapes.py::extract_torchvision_shapes`.

**Method C — Hardcoded** (fixed published architectures, e.g., DeepSpeech2)
Read a reference implementation and hardcode the known dimensions.
See `scripts/gen-definition/model_shapes.py::deepspeech2_shapes`.

---

### Convention: `model:xx` tag

Every definition derived from a real model carries one `model:` tag:
- `"model:qwen1.5-moe-a2.7b"` — lowercase, hyphens, matches HF repo name suffix
- `"model:olmoe-1b-7b"`, `"model:resnet50"`, `"model:mobilenetv3-large"`, `"model:deepspeech2"`

Manually authored definitions not tied to a model: omit the `model:` tag (just `"status:active"`).

---

### Convention: `baseline-solution:<framework>` tag

Every definition except `simd-loop` carries exactly one `baseline-solution:` tag,
naming which reference framework's kernel it should be benchmarked against for
speedup — independent of the `model:` tag, which only records shape provenance.

**Always ask the user which framework applies — do not infer it silently from the
`model:` tag or dtype.** Those correlate with the choice in today's definitions but
are not a reliable rule for a brand-new op_type or model the user hasn't described
yet. Use them only as a hint to make the question concrete, e.g.: "This looks like a
CV op captured from resnet50 — should the baseline be `ncnn`, or something else?"

For `simd-loop` definitions, skip the question — omit the tag entirely (they're
compared against `reference`/`autovec` solutions, not an external framework).

This tag currently records *intent* — as of this writing, `bench-trace/solutions/`
has no actual baseline solution for any op_type except a stale/dangling
`conv1d`/`deconv2d`/`deconv2d_depthwise` (ncnn), and there is no `ggml`/`llama.cpp`
`Builder` wired into `bench/compile/builders/` yet. Don't assume the tag implies a
solution already exists — check `bench-trace/solutions/` directly.

---

### Convention: dtype variants and INT8 naming

Definitions sharing the same op_type and const axes but different dtypes are
**separate definitions** with the quantization scheme encoded in the name:

| name pattern | dtype in inputs | scheme | output dtype |
|---|---|---|---|
| `gemm_fp32_n{N}_k{K}` | float32 | — | float32 |
| `gemm_q8_0_n{N}_k{K}` | int8 + float16 scales | Q8_0 (llama.cpp) | float32 |
| `gemm_w8a8ch_n{N}_k{K}` | int8 | ncnn per-channel | int8 |

**Q8_0 scheme** (llama.cpp, block size = 32):
- `A [M, K]` int8 + `A_scales [M, K_blk]` float16, where `K_blk = K // 32` (const axis)
- `B [N, K]` int8 + `B_scales [N, K_blk]` float16
- Output `C [M, N]` float32 (dequantized after accumulation, matching llama.cpp)
- K must be divisible by 32

**w8a8ch scheme** (ncnn per-channel):
- `A [M, K]` int8 + `input_scale` scalar float32 (per-tensor activation)
- `B [N, K]` int8 + `weight_scales [N]` float32 (per-channel weights)
- Output `C [M, N]` int8 with requantization

**MoE INT8 uses Q8_0** (llama.cpp is the MoE baseline; ncnn has no MoE):
- routing `router_weight [n_expert, n_embd]` stays float32 always
- expert weights and hidden_states quantized with Q8_0
- Scale shapes: `hs_scales [n_tokens, n_embd_blk]`, `gate_scales [n_expert, n_ff, n_embd_blk]`, etc.

For MHA: use canonical `[M, n_heads, head_dim]` layout. Baseline wrappers handle
framework-specific transposes — do not encode layout in the definition schema.

---

### Step 1: Survey existing definitions for context

```bash
ls bench-trace/definitions/
```

Check whether the op family already has a subdirectory, and read one or two existing
definitions in the closest analogous family to calibrate the format.

---

### Step 2: Elicit the design from the user

Ask the user to clarify any of these until they are all unambiguous:

1. The op's mathematical behavior (one sentence).
2. Which dimensions are fixed per kernel config (`const`) vs. per workload (`var`).
3. Concrete axis names (use the same casing conventions as existing defs: `C_in`, `H`,
   `N`, `Kh`, etc.).
4. Input tensor names, shapes, dtypes.
5. Output tensor name, shape, dtype.
6. How output shape axes relate to input + const axes (constraints).
7. Any scalar inputs (shape: null) and their typical values.
8. Which framework the `baseline-solution:` tag should name (e.g. `ncnn`,
   `llama.cpp`, or something else) — unless this is a `simd-loop` definition, in
   which case skip this and omit the tag.

---

### Step 3: Present the plan

Before writing any file, show the complete definition JSON with all fields filled in,
plus the output path. For example:

```
New definition: rms_norm_d768
  Output: bench-trace/definitions/rms_norm/rms_norm_d768.json
  Tags: status:active, baseline-solution:llama.cpp

  Axes:
    N (var)   — batch / token count
    D (const) — hidden dim = 768
  Inputs:
    x      [N, D]  float32
    weight [D]     float32
  Outputs:
    output [N, D]  float32
  Constraints:
    (none — output shape is fully determined by input shape)
  Reference:
    def run(x, weight): return (x / norm(x)) * weight

Proceed?
```

Wait for explicit user confirmation before writing.

---

### Step 4: Write the definition JSON

Use the Write tool to write to `bench-trace/definitions/<op_type>/<name>.json`.

Before writing, double-check:
- Every axis in any `shape` list appears in `"axes"`.
- Every axis referenced in `"constraints"` appears in `"axes"`.
- `const` axes have integer `value` fields.
- Exactly one `baseline-solution:<ncnn|llama.cpp>` tag is present, unless this is a
  `simd-loop` definition (omit it there).
- `var` axes that are derived have a constraint, and `"parent"` if grouped.
- The `reference` string is a valid Python function — mentally trace it for a simple
  input to confirm it returns the right shape and dtype.
- The `reference` string uses actual integer literals where const axis values appear,
  not symbolic axis names.

---

### Step 5: Verify TraceSet loads the definition

```bash
python -c "
from bench.data import TraceSet
ts = TraceSet.from_path('bench-trace')
d = ts.get_definition('<name>')
print('OK:', d.name, '| axes:', list(d.axes.keys()))
"
```

If this errors, diagnose and fix before declaring success.

---

### Step 6: Tell the user what's next

```
Definition written to bench-trace/definitions/<op_type>/<name>.json.
Add workloads next:
  /gen-workload — add workloads to <name>
```

---

## Common bugs

| Symptom | Root cause | Fix |
|---------|------------|-----|
| `ValidationError` on `axes` | Missing `value` on a const axis, or wrong `type` | Re-check each axis entry against the schema |
| `KeyError: '<name>'` from TraceSet | TraceSet requires a matching workload JSONL to recognize a definition | Create an empty `bench-trace/workloads/<op_type>/<name>.jsonl` (zero bytes is fine; `gen_workload.py` populates it) |
| `reference` raises `NameError` at eval time | Axis name used symbolically inside the Python string instead of its integer literal | Inline the const values as integer literals in the function body |
| Output shape mismatch at correctness check | Constraint wrong — off-by-one in floor-division or wrong formula | Re-derive the output size formula by hand; test with a small example |
| `parent` axis not a `var` | `parent` must reference another `var` axis, not a `const` one | Remove or fix the `parent` field |
| Tempted to guess `baseline-solution:` from the `model:` tag or dtype | Past definitions correlate CV/w8a8ch with ncnn and LLM/q8_0 with llama.cpp, but that's not a rule | Ask the user directly instead of inferring — the correlation is only a hint for framing the question |
