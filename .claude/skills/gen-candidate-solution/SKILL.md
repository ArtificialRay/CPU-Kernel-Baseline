---
name: gen-candidate-solution
description: Write a reference-scalar candidate solution (harness baked into its sources) and add it with kgen; use when adding a new op type or regenerating reference-scalar solutions
---

## gen_candidate_solution

A `reference-scalar` solution is the scalar, obviously-correct implementation of a definition:
the ground truth the evaluator and `kgen check` compare other kernels against, and the starting
file an LLM kernel replaces. Each solution JSON embeds its three sources; `CandidateBuilder`
compiles them with no separate harness directory.

There is no generator script and no template directory. You write the solution JSON (usually by
copying the closest existing one and changing what the definition changes), then:

| Command | Purpose |
|---|---|
| `kgen show <def>` | const/var axes, input dtypes, workloads, and which solutions exist |
| `kgen add solution FILE [--replace]` | lint FILE and write `bench-trace/solutions/<dataset>/<author>/<op_type>/<def>.json` |
| `kgen check <def>` | static lint always; on aarch64 also builds and runs reference-scalar against the reference |

---

## What to extract from the user's message

- **Op type / definition(s)** - ask if not stated. One definition, or every definition of the op type.
- **Dataset** - from the definition's `baseline-solution:<backend>` tag (`ncnn`, `llama.cpp`, `kleidiai`):
  the reference-scalar lives in the same dataset as its baseline, author from
  `config/kernel_contracts.yaml` (`reference_scalar_authors`: `reference-scalar`; simd-loop uses
  `reference`). **Check the tag; do not put an ncnn candidate under a llama.cpp-tagged definition.**
- **Variant** - plain float32 or quantized; look at `inputs[*].dtype`.

---

## What a solution JSON looks like

```json
{
  "name":        "reference-scalar_<def_name>",
  "definition":  "<def_name>",
  "dataset":     "<ncnn|llama.cpp|kleidiai>",
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
    { "path": "<op_type>.h",   "content": "// constexpr params + inner_<op_type> declaration" },
    { "path": "<op_type>.cpp", "content": "// armbench_entry_<op_type> -> inner_<op_type>" },
    { "path": "kernel.cpp",    "content": "// scalar inner_<op_type>" }
  ]
}
```

Constraints that lint or the build enforce:
- `name` must be `<author>_<definition>`; `definition` must already be in the warehouse.
- The source file names are always `<op_type>.h`, `<op_type>.cpp`, `kernel.cpp`, named after the
  definition's own `op_type` (a quantized variant still produces `conv2d.h`, not `conv2d_w8a8ch.h`).
- `entry_point` is `"<op_type>.cpp::armbench_entry_<op_type>"`, other format will cause a duplicate-symbol link error
- Entry points are `extern "C"`, not `static`. No OpenMP, threads, `fork`, they will be rejected by source pattern scanning at evaluation time.

Start from the closest existing file: `kgen show` lists solutions; open one under
`bench-trace/solutions/<dataset>/reference-scalar/<op_type>/` and copy its structure.

---

## Writing the sources

### 1. Classify axes and inputs

- **const axes** are baked into `<op_type>.h` as `constexpr int <name> = <value>;` inside
  `namespace <op_type>_def`, using the axis's own name.
- **var axes** vary per workload and are entry-function parameters only.
- **scalar inputs** (`"shape": null`) are baked as `constexpr` too, but only if every workload of the definition carries the same value (check the workload file); a scalar that varies per workload cannot be baked. A per-channel scale such as `weight_scales [C_out]` is a regular tensor passed as a pointer.
- **dtype**: every input's dtype decides its pointer type. Raw-ABI runners support the dtypes`bench/datasets/raw.py` maps (float32, int8, and whatever it lists now; an unsupported dtype raises at run time, never silently wrong data).

### 2. Entry signature = what the runner builds

`RawDataset.wrap_inputs` builds the call from the definition, dtype-aware:
1. **tensor pointers**: non-scalar inputs in `definition.inputs` order - primary input first,
   the output injected second, then the remaining tensors;
2. **var dims**: every var axis found in **any** tensor's shape (not just the primary one),
   deduplicated in first-seen order.

Get this order exactly right; a mismatch is a SIGSEGV with no useful message. Examples:

```c
// conv2d: weight's shape contributes the extra var axis C_out
extern "C" int armbench_entry_conv2d(const float* input, float* output, const float* weight,
                                     const float* bias, int N, int C_in, int H, int W, int C_out);
// packed ggml gemm: A bf16, B as uint8 blocks [N, K_bytes], output float
extern "C" int armbench_entry_gemm(const uint16_t* A, float* out, const uint8_t* B, int M);
```

### 3. The three files

- `<op_type>.h`: `#pragma once`, the `<op_type>_def` namespace with the constexpr values, and the `extern "C"` declaration of `inner_<op_type>`.
- `<op_type>.cpp`: `armbench_entry_<op_type>` computes derived dims from the constexprs and calls `inner_<op_type>`; returns 0.
- `kernel.cpp`: the scalar `inner_<op_type>` using the `_def` constants. This is the file an LLM replaces, so keep it plain loops.

For a quantized kernel follow the definition's own `reference` Python: typically `real = int8_value * scale` with `scale = input_scale * weight_scales[c]` (a dequantization multiplier, the opposite of ncnn's own int8 *quantization* scale fields). Accumulate in `int64_t`, apply
scale and bias in `double`, cast to `float` at the very end, to match the reference's float64 intermediates. For ggml block formats, decode the blocks exactly as ggml does.

---

## Workflow

1. `kgen show <def>`; check the dataset tag and which solutions exist.
2. Write `<def>.json` in the scratchpad (a throwaway snippet that renders it from the definition is fine for a whole op type; render, then add each file).
3. `python -m kgen add solution <file>` - problems come back as sentences naming the field;
   fix and rerun. `--replace` to overwrite.
4. `python -m kgen check <def> --static-only` anywhere.
5. On an aarch64 machine: `python -m kgen check <def> --num-selected-workloads 5`. The
   reference-scalar leg must pass; if the baseline leg also exists, the diagnosis line tells
   which of reference / reference-scalar / baseline is the likeliest culprit when they disagree
   (with stale calibration floors on a `correctness:sqnr` definition, recalibrate first).
6. `bench-trace/` is gitignored: sync it to the Graviton box yourself (`rsync -az --delete`
   of the solutions directory) before running the execution legs there.

---

## Common problems

| Symptom | Cause | Fix |
|---------|-------|-----|
| SIGSEGV / exit 139 | Entry arg order differs from `wrap_inputs` | Re-derive tensor order + var-dim order from the definition |
| Garbage on a quantized definition | Pointer type does not match the input dtype | Use the dtype's pointer type; check `raw.py`'s dtype map |
| `name: expected <author>_<definition>` | Wrong `name` | Make it `reference-scalar_<def>` |
| `definition ... is not in the warehouse` | Definition not added yet | `kgen add definition` first |
| `axis ... is not const` while baking | A var axis put in the constexpr block | Move it to the entry signature |
| Duplicate symbol at link | `entry_point` points at `kernel.cpp` | `<op_type>.cpp::armbench_entry_<op_type>` |
| Wrong backend gets a candidate | Ignored the `baseline-solution:` tag | Use the dataset that matches the tag |
