---
name: gen-candidate-solution
description: Add a new candidate solution with its harness baked into sources; use when adding a new op type or regenerating reference-scalar solutions
---

## gen_candidate_solution

Generate `reference-scalar` candidate solution JSONs for a given op type.
Each solution embeds all three source files directly; `CandidateBuilder` compiles
them without any separate harness directory.

Script: `scripts/gen_candidate_solution.py`

---

## What to extract from the user's message

Skills have no formal parameters — extract context from the user's natural language request:

- **Op type** — which operator to generate solutions for (e.g. `conv2d`, `conv1d`,
  `conv2d_depthwise`, `deconv2d`, `deconv2d_depthwise`). If not mentioned, ask before proceeding.
- **Scope** — a specific definition name to target, or all definitions in the op type (default).
- **Mode** — generating from scratch (new op type) or regenerating existing solutions.

Valid invocations look like:
```
Generate reference-scalar solutions for conv2d
Regenerate all deconv2d candidate solutions
Add a candidate solution for conv2d_kh3_kw3_sh1_sw1_dh1_dw1_c64_c64
Add candidate solutions for the new conv1d op type
```

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
  "description": "Scalar raw-float* <op_type> for <def_name>. Constexpr-baked dims; armbench_entry_<op_type> calls inner_<op_type>. Ground-truth correctness baseline.",
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
- `entry_point` must be `"<op_type>.cpp::armbench_entry_<op_type>"`, not `"kernel.cpp::inner_<op_type>"` —
  pointing at `kernel.cpp` causes a duplicate symbol link error
- The three source files must be present; `CandidateBuilder` adds the solution's own dir to `-I`

---

## Workflow: regenerate existing op type

If the op type is already registered in `op_config.json` and templates already exist:

```bash
# All definitions
python -m scripts.gen_candidate_solution --op-type <op_type>

# Single definition
python -m scripts.gen_candidate_solution --op-type <op_type> --definition <def_name>

# Sync to Graviton (bench-trace/ is gitignored — must rsync manually)
rsync -az --delete \
    bench-trace/solutions/ncnn/reference-scalar/<op_type>/ \
    ubuntu@<host>:arm-bench/bench-trace/solutions/ncnn/reference-scalar/<op_type>/

# Validate on remote
ssh ubuntu@<host> "cd arm-bench && python3 -m scripts.eval_candidates 2>&1"
```

---

## Workflow: add a new op type

### Step 1: Classify the definition's axes

Open a sample definition JSON from `bench-trace/definitions/<defs_family>/`.

- **const axes** (`"type": "const"`): fixed across all workloads → baked into `<op_type>.h` as constexpr
- **var axes** (`"type": "var"`): vary per workload → entry parameter list only
- **scalar_inputs**: not in the definition axes; read from the first workload in
  `bench-trace/traces.golden/<op_type>/` — check the workload's `inputs` dict for `"type": "scalar"` entries

Example: for conv2d, const axes are `C_in, C_out, Kh, Kw, Sh, Sw, Dh, Dw`; var axes are `N, H, W`;
scalar_inputs are `pad_top, pad_left, activation_type`.

---

### Step 2: Design the ABI

`bench/datasets/raw.py` `wrap_inputs` builds the entry args automatically from the definition:
1. **Tensor pointers**: non-scalar inputs in `definition.inputs` order — primary input first,
   output injected second, then remaining tensors
2. **Var dims**: from the primary input's shape template, axes where `type == "var"`

Verify your C entry signature matches this order exactly. Mismatches produce SIGSEGV with no
useful error.

Example for conv2d (primary input shape `[N, C_in, H, W]`, `C_in` const, `N H W` var):
```c
extern "C" int armbench_entry_conv2d(
    const float* input, float* output, const float* weight,
    int N, int H, int W)
```

---

### Step 3: Write the three template files

Location: `scripts/candidate_binding_templates/`. Name them `<op_type>.h.tmpl`,
`<op_type>.cpp.tmpl`, and either `<op_type>.kernel.cpp.tmpl` or `kernel.cpp.tmpl`
(the latter as a shared fallback).

**`<op_type>.h.tmpl`** — constexpr namespace + `inner_<op_type>` declaration.
`{{Cin}}`, `{{Kh}}`, etc. are substitution tokens: `gen_candidate_solution.py` replaces
each `{{Key}}` with the integer value resolved from `const_axes` or `scalar_inputs` in
`op_config.json`, producing the final C++ source baked into the solution JSON:
```cpp
// Auto-generated — do not hand-edit.
#pragma once
namespace <op_type>_def {
constexpr int Cin = {{Cin}};   // → e.g. 64
constexpr int Kh  = {{Kh}};   // → e.g. 3
// ... one line per const param
} // namespace <op_type>_def

#ifdef __cplusplus
extern "C" {
#endif
void inner_<op_type>(...);
#ifdef __cplusplus
}
#endif
```

**`<op_type>.cpp.tmpl`** — binding harness; usually no `{{}}` placeholders:
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

**`<op_type>.kernel.cpp.tmpl`** — reference-scalar; LLM replacement target:
```cpp
#include "<op_type>.h"
using namespace <op_type>_def;

extern "C" void inner_<op_type>(...)
{
    // scalar loops using <op_type>_def:: constexpr constants
}
```

---

### Step 4: Register the op type in op_config.json

Add an entry to `scripts/candidate_binding_templates/op_config.json`:

```json
{
  "<op_type>": {
    "const_axes": {
      "Cin": "C_in",
      "Cout": "C_out"
    },
    "name_extract_axes": {
      "pad": "_p([0-9]+)_"
    },
    "scalar_inputs": ["pad_top", "pad_left", "activation_type"],
    "defs_family":   "<subdirectory under bench-trace/definitions/>",
    "traces_subdir": "<op_type>"
  }
}
```

Fields:
- `const_axes`: maps each `{{placeholder}}` in `*.h.tmpl` → axis name in the definition JSON
- `name_extract_axes`: extract values from the definition name via regex (group 1 = int value);
  omit this key if not needed
- `scalar_inputs`: keys to read from the first workload's scalar inputs; use `[]` if none
- `defs_family`: subdirectory under `bench-trace/definitions/`
- `traces_subdir`: subdirectory under `bench-trace/traces.golden/`

---

### Step 5: Run codegen and verify

```bash
python -m scripts.gen_candidate_solution --op-type <op_type>

rsync -az --delete \
    bench-trace/solutions/ncnn/reference-scalar/<op_type>/ \
    ubuntu@<host>:arm-bench/bench-trace/solutions/ncnn/reference-scalar/<op_type>/

ssh ubuntu@<host> "cd arm-bench && python3 -m scripts.eval_candidates 2>&1"
```

---

## Common bugs

| Symptom | Root cause | Fix |
|---------|-----------|-----|
| SIGSEGV / exit 139 | C entry arg order doesn't match what `raw.py` `wrap_inputs` builds | Trace through `wrap_inputs`: tensor order + var dim order must match C entry exactly |
| `FileNotFoundError: Golden trace not found` | `traces.golden/<op_type>/` not populated | Run `scripts/gen_definitions.py` first |
| `axis ... is not const` | A dimension put in constexpr block is actually var for this definition | Move it to the entry signature instead |
| `FileNotFoundError: Template not found` | Missing one of the three `.tmpl` files | Add it to `scripts/candidate_binding_templates/` |
| Duplicate symbol link error | `entry_point` set to `kernel.cpp::inner_<op_type>` | Change to `<op_type>.cpp::armbench_entry_<op_type>` |
| `name_extract_axes pattern ... found no match` | Regex doesn't match the definition naming convention | Check `bench-trace/definitions/<defs_family>/` filenames and adjust the pattern |
