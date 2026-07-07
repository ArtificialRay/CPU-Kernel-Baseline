---
name: gen-candidate-solution
description: Add a new candidate solution with its harness baked into sources; use when adding a new op type or regenerating reference-scalar solutions
---

## gen_candidate_solution

Generate `reference-scalar` candidate solution JSONs for a given op type.
Each solution embeds all three source files directly; `CandidateBuilder` compiles
them without any separate harness directory.

Script: `scripts/gen_candidate_solution.py`

No config file drives this — everything is derived straight from the Definition JSON
(mirrors `scripts/gen_baseline_solution.py`'s design). There is no
`op_config.json` anymore; don't recreate one.

---

## What to extract from the user's message

Skills have no formal parameters — extract context from the user's natural language request:

- **Op type** — which operator to generate solutions for (e.g. `conv2d`, `conv2d_depthwise`, `gemm`,
  `pooling`, `lstm`). If not mentioned, ask before proceeding.
- **Scope** — a specific definition name to target, or all definitions in the op type (default).
- **Variant** — plain float32, or a quantized variant (e.g. `w8a8ch`). Check the definition's
  `inputs[*].dtype` — see Step 0 below.
- **Mode** — generating from scratch (new op type / new variant) or regenerating existing solutions.

Valid invocations look like:
```
Generate reference-scalar solutions for conv2d
Regenerate all conv2d_depthwise candidate solutions
Add a candidate solution for conv2d_kh3_kw3_sh1_sw1_dh1_dw1_c64_c64
Add w8a8ch candidate solutions for conv2d
```

---

## Step 0: Check the definition's tag — which backend does it belong to?

`bench-trace/definitions/<op_type>/*.json` can mix definitions meant for **different backends**,
distinguished by a `baseline-solution:<backend>` tag, e.g.:
```json
"tags": ["status:active", "model:olmoe-1b-7b", "baseline-solution:llama.cpp"]
```
vs.
```json
"tags": ["status:active", "model:mobilenetv3-large", "baseline-solution:ncnn"]
```
`gen_candidate_solution.py` only ever writes to `bench-trace/solutions/ncnn/reference-scalar/` — so
it defaults to processing **only `baseline-solution:ncnn`-tagged** definitions (`--tag`, default
`baseline-solution:ncnn`) when generating a whole op type. A `baseline-solution:llama.cpp`-tagged
definition under the same op_type directory must NOT get an ncnn-routed reference-scalar candidate
here — that belongs to a different, not-yet-built candidate track. `--tag` filtering is skipped when
`--definition` is given explicitly (an explicit ask always goes through), so double-check the tag
yourself before generating a single definition by name.

`gemm`'s definitions are a real example of this split: `gemm_fp32_n1000_k1280` etc. and the
`w8a8ch` variants are `baseline-solution:ncnn`; `gemm_fp32_n1024_k2048` etc. and the `q8_0` variants
are `baseline-solution:llama.cpp`.

---

## What a solution JSON looks like

Every candidate solution is a single JSON file at:
```
bench-trace/solutions/ncnn/reference-scalar/<op_type>/<def_name>.json
```

Example for `conv2d`:
```json
{
  "name":        "reference-scalar_<def_name>",
  "definition":  "<def_name>",
  "dataset":     "ncnn",
  "author":      "reference-scalar",
  "description": "Scalar raw-pointer <op_type> for <def_name>. Constexpr-baked dims; armbench_entry_<op_type> calls inner_<op_type>. Ground-truth correctness baseline.",
  "spec": {
    "language":        "cpp",
    "target_hardware": ["graviton3", "aarch64-sve"],
    "entry_point":     "<op_type>.cpp::armbench_entry_<op_type>",
    "dependencies":    [],
    "isa_features":    [],
    "compile_flags":   ["-O2", "-std=c++14"],
    "link_flags":      []
  },
  "sources": [
    { "path": "<op_type>.h",   "content": "// constexpr params + inner_<op_type> declaration..." },
    { "path": "<op_type>.cpp", "content": "// binding harness: armbench_entry_<op_type> → inner_<op_type>..." },
    { "path": "kernel.cpp",    "content": "// reference-scalar inner_<op_type> implementation..." }
  ]
}
```

Key constraints:
- `dataset` must be `"ncnn"` — `CandidateBuilder` handles all non-baseline non-simd-loop solutions
- Output source filenames (`<op_type>.h`/`<op_type>.cpp`) are always named after the definition's
  own `op_type`, **never** the template prefix — a quantized variant like `conv2d_w8a8ch` still
  produces `conv2d.h`/`conv2d.cpp`, because `entry_point` and the `armbench_entry_<op_type>` symbol
  raw.py's runner looks up are keyed by `op_type`, not by which variant produced the source
- `entry_point` must be `"<op_type>.cpp::armbench_entry_<op_type>"`, not `"kernel.cpp::inner_<op_type>"` —
  pointing at `kernel.cpp` causes a duplicate symbol link error
- The three source files must be present; `CandidateBuilder` adds the solution's own dir to `-I`

---

## Workflow: regenerate existing op type

If templates already exist for this op type/variant:

```bash
# All ncnn-tagged definitions for this op type, fp32 templates only
python -m scripts.gen_candidate_solution --op-type <op_type>

# All ncnn-tagged definitions, including a quantized variant
python -m scripts.gen_candidate_solution --op-type <op_type> --variant-quant-suffix w8a8ch

# Single definition (bypasses --tag filtering — you're asking for it by name)
python -m scripts.gen_candidate_solution --op-type <op_type> --definition <def_name> \
    [--variant-quant-suffix w8a8ch]

# A definition whose axes don't match its op_type's usual shape at all (e.g. pooling's
# global-average-pooling definition — no Kh/Kw/stride/pad axes):
python -m scripts.gen_candidate_solution --op-type pooling --definition pooling_fp32_global_avg \
    --template-prefix pooling_global_avg

# Sync to Graviton (bench-trace/ is gitignored — must rsync manually)
rsync -az --delete \
    bench-trace/solutions/ncnn/reference-scalar/<op_type>/ \
    ubuntu@<host>:arm-bench/bench-trace/solutions/ncnn/reference-scalar/<op_type>/

# Validate on remote
ssh ubuntu@<host> "cd arm-bench && python3 -m scripts.eval_candidates --op <op_type> 2>&1"
```

---

## Workflow: add a new op type (or a new quantized variant of an existing one)

### Step 1: Classify the definition's axes and inputs

Open a sample definition JSON from `bench-trace/definitions/<op_type>/`.

- **const axes** (`"type": "const"`): fixed across all workloads → baked into `<op_type>.h` as
  `constexpr <name> = {{name}};` — the placeholder token is always the axis's **own name**, there's
  no renaming/aliasing step anymore
- **var axes** (`"type": "var"`): vary per workload → entry parameter list only
- **scalar inputs** (an entry in `definition.inputs` with `"shape": null`, not an axis): baked as a
  `constexpr` the same way, by its own name, **if and only if** it's the same value across every
  workload for that definition (checked automatically against
  `bench-trace/workloads/<op_type>/<def_name>.jsonl` — a scalar that genuinely varies per workload
  can't be represented this way and the definition is skipped with a clear message). A
  per-output-channel scale (`"shape": ["C_out"]`, dtype float32) is NOT a scalar input — it's a
  regular tensor, passed as a runtime float pointer, no baking needed.
- **dtype**: check every `inputs[*].dtype`. All-`float32` → fp32 template variant. Any non-float32
  (e.g. `int8` for `w8a8ch`) → quantized template variant, needs `--variant-quant-suffix`.

Example: for `conv2d_w8a8ch`, const axes are `Kh, Kw, Sh, Sw, Dh, Dw, pad_top, pad_left`; var axes
are `N, C_in, H, W, C_out`; `input_scale` is a scalar (constant across workloads, bake it);
`weight_scales` (shape `[C_out]`) is a real tensor, not baked.

---

### Step 2: Design the ABI

`bench/datasets/raw.py`'s `RawDataset.wrap_inputs` builds the entry args automatically from the
definition — **dtype-aware** (each tensor is cast/pointer-typed per its own declared `dtype`;
currently `float32` and `int8` are supported, anything else raises `NotImplementedError` at run
time, not silently wrong data):
1. **Tensor pointers**: non-scalar inputs (`shape != null`) in `definition.inputs` order — primary
   input first, output injected second (always `float32`, since every current Definition's output
   is), then remaining tensors, each using its own dtype's pointer type
2. **Var dims**: scans **every** tensor's shape template (not just the primary one), dedup'd by
   axis name in first-seen order — an axis that only appears in a non-primary tensor (like
   `conv2d`'s `C_out`, which lives in `weight`'s shape, not `input`'s) is still picked up

Verify your C entry signature matches this order exactly. Mismatches produce SIGSEGV with no
useful error.

Example for plain `conv2d` (`input` is primary, shape `[N, C_in, H, W]`; `weight` shape
`[C_out, C_in, Kh, Kw]` contributes the new var axis `C_out`):
```c
extern "C" int armbench_entry_conv2d(
    const float* input, float* output, const float* weight, const float* bias,
    int N, int C_in, int H, int W, int C_out)
```
Example for `conv2d_w8a8ch` (same var-dim order; `input`/`weight` are now `int8_t*`, plus a new
`weight_scales` tensor pointer per `definition.inputs` order):
```c
extern "C" int armbench_entry_conv2d(
    const int8_t* input, float* output,
    const int8_t* weight, const float* bias, const float* weight_scales,
    int N, int C_in, int H, int W, int C_out)
```

---

### Step 3: Write the three template files

Location: `scripts/candidate_binding_templates/`. Name them `<prefix>.h.tmpl`, `<prefix>.cpp.tmpl`,
and either `<prefix>.kernel.cpp.tmpl` or `kernel.cpp.tmpl` (shared fallback).

`<prefix>` is `<op_type>` for the plain/fp32 variant, or `<op_type>_<quant-suffix>` for a quantized
variant (e.g. `conv2d_w8a8ch`) — **the prefix only selects which template file gets read off disk.**
The rendered output is always written as `<op_type>.h` / `<op_type>.cpp` / `kernel.cpp` regardless
of prefix, so the symbol names inside the template content must use the **base op_type**
(`armbench_entry_conv2d`, `inner_conv2d`, namespace `conv2d_def`), even inside a `conv2d_w8a8ch.*.tmpl`
file — never the prefix.

**`<prefix>.h.tmpl`** — constexpr namespace + `inner_<op_type>` declaration. `{{Kh}}`, `{{input_scale}}`,
etc. are substitution tokens, one per const axis / scalar input, named exactly like the axis/input
itself (no config-driven renaming):
```cpp
#pragma once
namespace <op_type>_def {
constexpr int Kh  = {{Kh}};   // → e.g. 3
constexpr float input_scale = {{input_scale}};  // only for a variant that has a scalar input
} // namespace <op_type>_def

#ifdef __cplusplus
extern "C" {
#endif
void inner_<op_type>(...);
#ifdef __cplusplus
}
#endif
```

**`<prefix>.cpp.tmpl`** — binding harness; usually no `{{}}` placeholders:
```cpp
#include "<op_type>.h"
using namespace <op_type>_def;

extern "C" int armbench_entry_<op_type>(...var dims only...)
{
    // compute derived output dims from constexpr params
    inner_<op_type>(...);
    return 0;
}
```

**`<prefix>.kernel.cpp.tmpl`** — reference-scalar; LLM replacement target:
```cpp
#include "<op_type>.h"
using namespace <op_type>_def;

extern "C" void inner_<op_type>(...)
{
    // scalar loops using <op_type>_def:: constexpr constants
}
```

For a quantized kernel, match the dequant formula used by the corresponding baseline template
(`scripts/baseline_binding_templates/<prefix>.ncnn_kernel.cpp.tmpl`, if one exists) and the
Definition's own `"reference"` Python — typically `real = int8_value * scale` where
`scale = input_scale * weight_scales[c]` (a **dequantization** multiplier, the opposite convention
from ncnn's own int8 *quantization* scale fields — don't just copy ncnn's scale handling
uninverted). Accumulate in `int64_t`, do the scale multiply + bias add in `double`, cast to `float`
only at the very end, matching the Python reference's `float64` intermediate precision.

---

### Step 4: Generate and verify

```bash
python -m scripts.gen_candidate_solution --op-type <op_type> [--variant-quant-suffix <suffix>] \
    [--definition <def_name>] --dry-run   # check it resolves the prefix + renders cleanly first

python -m scripts.gen_candidate_solution --op-type <op_type> [--variant-quant-suffix <suffix>]

rsync -az --delete \
    bench-trace/solutions/ncnn/reference-scalar/<op_type>/ \
    ubuntu@<host>:arm-bench/bench-trace/solutions/ncnn/reference-scalar/<op_type>/

ssh ubuntu@<host> "cd arm-bench && python3 -m scripts.eval_candidates --op <op_type> 2>&1"
```

Note `scripts/eval_candidates.py`'s final `"N/M solutions fully passed"` summary line is unreliable
when using `--op`/`--definition` — it counts filtered-out solutions as passed. Trust the per-line
`[OK]`/`[FAIL]` output instead.

---

## Common bugs

| Symptom | Root cause | Fix |
|---------|-----------|-----|
| SIGSEGV / exit 139 | C entry arg order doesn't match what `raw.py` `wrap_inputs` builds | Trace through `wrap_inputs`: tensor order + var dim order must match C entry exactly |
| SIGSEGV / garbage output on a quantized definition | Tensor dtype not in `raw.py`'s `_DTYPE_TO_NP_AND_PTR` map, or candidate signature uses `float*` for what's actually `int8_t*` data | Check `definition.inputs[*].dtype`; add the dtype to the map if genuinely new (currently only `float32`/`int8` supported) |
| `axis ... is not const` | A dimension put in constexpr block is actually var for this definition | Move it to the entry signature instead |
| `has non-float32 input(s) and no --variant-quant-suffix given` (SKIP, not error) | Definition needs a quantized template set you haven't written/passed yet | Write `<op_type>_<suffix>.*.tmpl` and pass `--variant-quant-suffix <suffix>` |
| `... has unresolved placeholder(s) [...]` (SKIP, not error) | Definition's axes/inputs don't match the chosen template (e.g. pooling's global-avg definition has no Kh/Kw at all) | Use `--template-prefix` to force a different, purpose-built template set for that definition |
| `FileNotFoundError: Template not found` | Missing one of the three `.tmpl` files for the resolved prefix | Add it to `scripts/candidate_binding_templates/` |
| Duplicate symbol link error | `entry_point` set to `kernel.cpp::inner_<op_type>` | Change to `<op_type>.cpp::armbench_entry_<op_type>` |
| Wrong backend gets a candidate (e.g. an ncnn reference-scalar generated for a llama.cpp-tagged definition) | Ran with `--definition` (bypasses `--tag` filtering) without checking the definition's `baseline-solution:<backend>` tag first | Check the tag before using `--definition`; the no-`--definition` path already filters by `--tag` (default `baseline-solution:ncnn`) |
