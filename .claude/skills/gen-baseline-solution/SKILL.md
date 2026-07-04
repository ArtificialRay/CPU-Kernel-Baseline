---
name: gen-baseline-solution
description: Generate or regenerate baseline solution JSONs (ncnn today, other backends later) for an op type; use when adding a new op type baseline or after definitions are regenerated
---

## gen_baseline_solution

Generate baseline solution JSONs for a given op type and backend. Each solution embeds
all three source files directly: `{op_type}.h` (the harness contract), `binding.cpp`
(`armbench_entry_<op_type>`, a void*-ABI shim with per-definition params baked as
constexpr), and `kernel.cpp` (delegates to the backend library — `ncnn::*_arm` layers
linked from `libncnn.a` today). `NcnnBuilder` compiles these against the full ncnn
static lib for the `ncnn` backend.

Script: `scripts/gen_baseline_solution.py`

**Backend identity is a set of CLI parameters, not a hardcoded table.** `--tag` (which
definitions to pick up), `--author` (solution author name), `--template-dir` (where the
3 template files per op_type live), `--dataset` (solution `dataset` field), `--compile-flags`,
`--link-flags` all default to today's only backend (`ncnn`), but a future backend (e.g.
`llama.cpp`, for the `moe`/`rms_norm`/`mha`/quantized-`gemm` definitions already tagged
`baseline-solution:llama.cpp`) passes its own values explicitly — **don't add a new script
or a hardcoded backend table for it**; this script is the one place all backends' baseline
generation goes through.

---

## What to extract from the user's message

Skills have no formal parameters — extract context from the user's natural language request:

- **Op type** — which operator to generate baselines for (e.g. `conv2d`, `conv2d_depthwise`,
  `gemm`, `pooling`, `lstm`). If not mentioned, ask before proceeding.
- **Backend** — which library backs the kernel (`ncnn` unless the user names another, e.g.
  `llama.cpp`). If it's not `ncnn` and templates/CLI values for that backend don't exist yet,
  that's new-backend setup, not just a new op type — confirm scope with the user.
- **Scope** — a specific definition name to target, or all `--tag`-matching definitions in
  the op type (default).
- **Mode** — generating from scratch (new op type/backend) or regenerating after a
  definitions refresh.

Valid invocations look like:
```
Generate baseline-ncnn-arm solutions for conv2d, conv2d_depthwise, gemm, pooling, lstm
Regenerate conv2d baselines after the definitions were updated
Add baseline solutions for the new gemm op type
```

---

## What a baseline solution JSON looks like

Every baseline solution is a single JSON file at:
```
bench-trace/solutions/<dataset>/<author>/<op_type>/<def_name>.json
```
(`<dataset>`/`<author>` default to `ncnn`/`baseline-ncnn-arm` — see `--dataset`/`--author`).

Example for `conv2d`:
```json
{
  "name":        "baseline-ncnn-arm_<def_name>",
  "definition":  "<def_name>",
  "dataset":     "ncnn",
  "author":      "baseline-ncnn-arm",
  "description": "baseline-ncnn-arm baseline for <def_name>. binding.cpp bakes constexpr params and implements armbench_entry_<op_type> with a void* Mat ABI; kernel.cpp delegates to the backend library. Timing baseline for speedup computation.",
  "spec": {
    "language":        "cpp",
    "target_hardware": ["graviton3", "aarch64-sve", "graviton4", "aarch64-sve2"],
    "entry_point":     "binding.cpp::armbench_entry_<op_type>",
    "dependencies":    [],
    "isa_features":    [],
    "compile_flags":   ["-O3", "-std=c++17"],
    "link_flags":      ["-fopenmp"]
  },
  "sources": [
    { "path": "<op_type>.h",   "content": "// harness contract: ncnn::<op_type>_kernel(...) signature..." },
    { "path": "binding.cpp",   "content": "// armbench_entry_<op_type>: void* Mat ABI, computes runtime dims, pre-allocates output, calls the contract fn..." },
    { "path": "kernel.cpp",    "content": "// implements the contract fn using ncnn::*_arm (links against libncnn.a)..." }
  ]
}
```

Key constraints:
- `NcnnBuilder` (for the `ncnn` backend) is activated by `is_baseline=True` AND
  `solution.dataset == "ncnn"` — keep `--dataset ncnn` unless you're building a different
  backend's own builder too.
- `Benchmark._is_baseline()` checks `solution.author == config.baseline_author` — the author
  you generate with must match whatever `--baseline-author` the benchmark run uses.
- `entry_point` is always `"binding.cpp::armbench_entry_<op_type>"` — pointing at `kernel.cpp`
  causes a duplicate symbol link error (kernel.cpp defines the contract fn, not the entry).
- ncnn's `compile_flags` must include `-std=c++17` (ncnn headers require it) and `link_flags`
  must include `-fopenmp` (`libncnn.a` references OpenMP symbols) — both are defaults already.

---

## The void* Mat ABI (ncnn backend) — how args reach binding.cpp

`bench/datasets/ncnn.py`'s `NcnnDataset.wrap_inputs` builds the call args generically from
`definition.inputs`, N-ary, no padding: tensor 0 (the "primary" tensor, e.g. `input`/`A`/`x`)
gets shaped by its own rank (3D for a 4D-with-N=1/3D array, 2D for a 2D array), every other
tensor is passed flat 1D (matching how ncnn layers store `weight_data`/`bias_data` etc.).
`armbench_entry_<op_type>`'s signature must be exactly `(primary_v, top_v, *rest_v, opt_v)` —
however many real tensors the definition declares, no extra padding slots. Concretely:
- **conv2d/conv2d_depthwise** (`input, weight, bias`): `(bottom_v, top_v, weight_v, bias_v, opt_v)`.
- **gemm** (`A, B`): `(A_v, top_v, B_v, opt_v)` — `A` arrives as a genuine 2D Mat (`w=K,h=M`).
- **pooling** (`input` only): `(bottom_v, top_v, opt_v)`.
- **lstm** (`x, h0, c0, W_ih, W_hh, b`): `(x_v, top_v, h0_v, c0_v, Wih_v, Whh_v, b_v, opt_v)`.

binding.cpp always: reinterprets each `void*` as `const ncnn::Mat*`/`ncnn::Mat*`, computes any
runtime-varying output dims (e.g. `C_out` from `weight.total()`, not baked — **only bake a dim
as `{{placeholder}}` constexpr if the definition's axis is actually `"type": "const"`; a `"var"`
axis must be derived at runtime from a Mat's shape**), pre-allocates `top` via `.create(...)`,
then calls the op's contract function (declared in `{op_type}.h`, implemented in `kernel.cpp`).

---

## Workflow: regenerate existing op type

If the op type is already registered in `op_config.json` and templates already exist:

```bash
# All --tag-matching definitions for one op type (default --tag is baseline-solution:ncnn)
python -m scripts.gen_baseline_solution --op-type <op_type>

# Single definition
python -m scripts.gen_baseline_solution --op-type <op_type> --definition <def_name>

# All op types tagged for the ncnn backend
for op in conv2d conv2d_depthwise gemm pooling lstm; do
    python -m scripts.gen_baseline_solution --op-type $op
done

# A different backend supplies its own identity explicitly (no script/table change):
python -m scripts.gen_baseline_solution --op-type gemm \
    --tag baseline-solution:llama.cpp --author baseline-llama-cpp-arm \
    --template-dir scripts/llama_cpp_binding_templates --dataset llama_cpp \
    --compile-flags -O3 -std=c++17 --link-flags -lggml

# Sync to Graviton (bench-trace/ is gitignored — must rsync manually)
rsync -az --delete \
    bench-trace/solutions/ncnn/baseline-ncnn-arm/ \
    ubuntu@<host>:arm-bench/bench-trace/solutions/ncnn/baseline-ncnn-arm/

# Collect baselines on remote to verify they compile and run correctly
ssh ubuntu@<host> "cd arm-bench && python3 -m bench.cli collect-baselines --baseline-author baseline-ncnn-arm 2>&1"
```

A definition is silently **skipped** (not an error) if it doesn't carry `--tag`, if any input
dtype isn't `float32` (quantized baselines aren't supported yet), or if it's missing an axis
this op_type's `const_axes` expects (e.g. global-average-pooling defs have no `Kh`/`Kw` at
all — they need a different kernel shape, not this op_type's template).

---

## Workflow: add a new op type

### Step 1: Register const_axes in op_config.json

Add an entry to `scripts/candidate_binding_templates/op_config.json`:
```json
"pooling": {
  "const_axes": { "Kh": "Kh", "Kw": "Kw", "Sh": "Sh", "Sw": "Sw", "pad_top": "pad_top", "pad_left": "pad_left" },
  "defs_family": "pooling"
}
```
Only include axes that are actually `"type": "const"` in the definition — a `"var"` axis
(e.g. `C_out` for conv2d, `M` for gemm, `T` for lstm) must never be baked; derive it at
runtime in binding.cpp from a Mat's shape instead (see the ABI section above).

---

### Step 2: Understand the ncnn layer API for this op type

Identify which `ncnn::*_arm` class to use — **read the real headers under
`/home/rthu/l3/CPU-Kernel-Baseline/ncnn/src/layer/{arm/,}<layer>.{h,cpp}` (or wherever
the local ncnn checkout lives) rather than assuming API shape from another op's pattern**;
several fields are *not* default-initialized by these classes' constructors and silently
read as garbage if you skip them (confirmed for `InnerProduct_arm`, `Pooling_arm`, and
`LSTM_arm` — each has fields the constructor never touches).

| Op type          | ncnn class                          | Header                          | Gotchas |
|-------------------|--------------------------------------|----------------------------------|---------|
| conv2d            | `ncnn::Convolution_arm`              | `convolution_arm.h`               | `C_out` is often a var axis — derive from `weight.total()`, don't bake |
| conv2d_depthwise  | `ncnn::ConvolutionDepthWise_arm`     | `convolutiondepthwise_arm.h`       | set `group = C`, `num_output = C` |
| gemm              | `ncnn::InnerProduct_arm`             | `innerproduct_arm.h`              | pass `bottom_blob` as a genuine 2D Mat (`dims==2`) to get the one-call `(M,K)->(M,N)` gemm path; `weight_data` stays flat (reshaped internally) |
| pooling           | `ncnn::Pooling_arm`                  | `pooling_arm.h`                   | set **every** field explicitly (`pooling_type`, `kernel_w/h`, `stride_w/h`, `pad_*`, `global_pooling`, `pad_mode`, `avgpool_count_include_pad`, `adaptive_pooling`, `out_w/h`) — none are default-initialized; use `pad_mode=1` ("valid") to match a plain floor-based output-size formula |
| lstm              | `ncnn::LSTM_arm`                     | `lstm_arm.h`                      | ncnn's internal gate order is **I,F,O,G**, not PyTorch's **I,F,G,O** — swap chunks 2/3 when building `weight_xc_data`/`weight_hc_data`/`bias_c_data`; those must be real 3D-shaped Mats (`create_pipeline` reads them via `.channel()/.row()`), not flat; use the 3-blob `forward(bottom_blobs={x,h0,c0}, top_blobs, opt)` overload to honor given initial state (the 1-Mat overload always zero-inits) |

Common settings across all of these:
- `opt.use_packing_layout = false`, `opt.num_threads = 1` (already ncnn's baseline `Option`
  defaults elsewhere in the harness)
- `int8_scale_term = 0`, `activation_type = 0`, `activation_params = Mat()` unless the
  definition's reference implementation actually fuses an activation
- `create_pipeline(opt)` before `forward(...)` — required for every layer even when it looks
  like a no-op (e.g. `Pooling_arm`'s is a no-op for non-adaptive pooling, but still call it)

---

### Step 3: Write the 3 template files

Location: `scripts/baseline_binding_templates/<op_type>.{ncnn_contract.h,binding.cpp,ncnn_kernel.cpp}.tmpl`.
Use `scripts/baseline_binding_templates/gemm.*.tmpl` or `pooling.*.tmpl` as the reference
pattern (both self-contained, void*-Mat-ABI, runtime-derived var dims) rather than any
older `inner_<op_type>(float* ...)`-style code — that raw-pointer style predates the current
Mat-ABI bridge and no longer matches what `bench/datasets/ncnn.py` actually sends.

`{{placeholder}}` substitution works the same as the harness templates:
`gen_baseline_solution.py` fills each `{{key}}` from `op_config.json`'s `const_axes` (or
`name_extract_axes`, if the definition encodes a param only in its name).

**`ncnn::Mat` copying rules** (defensive, matches `bench/datasets/_ncnn_lib/_mat_factory.cpp`):
- Never bulk-`memcpy` a whole multi-row/channel Mat in one call — rows/channels can have
  alignment padding (`cstep`). Copy row-by-row (`.row(i)`) for 2D Mats, channel-by-channel
  (`.channel(c)`) for 3D Mats.
- A flat 1D Mat's `.data` is a plain contiguous `float*` — safe to index directly (used for
  reshaping lstm's flat weight inputs into their proper 3D shape).

---

### Step 4: Run codegen and verify

```bash
# Dry-run to preview
python -m scripts.gen_baseline_solution --op-type <op_type> --dry-run

# Write solutions
python -m scripts.gen_baseline_solution --op-type <op_type>

# Verify TraceSet loads them without errors
python -c "
from bench.data import TraceSet
ts = TraceSet.from_path('bench-trace')
for name, sols in ts.solutions.items():
    baseline = [s for s in sols if s.author == 'baseline-ncnn-arm']
    if baseline:
        print('OK', name, '->', baseline[0].name)
"

# On Graviton: collect baselines to confirm compile + correctness
ssh ubuntu@<host> "cd arm-bench && python3 -m bench.cli collect-baselines --baseline-author baseline-ncnn-arm 2>&1"
```

---

## Template vs harness file locations

| File type                              | Directory                                | Script that reads it            |
|-----------------------------------------|-------------------------------------------|----------------------------------|
| `op_config.json`                        | `scripts/candidate_binding_templates/`    | both gen scripts                 |
| `<op_type>.{h,cpp,kernel.cpp}.tmpl`     | `scripts/candidate_binding_templates/`    | gen_candidate_solution only      |
| `<op_type>.ncnn_contract.h.tmpl`        | `--template-dir` (default `scripts/baseline_binding_templates/`) | gen_baseline_solution only |
| `<op_type>.binding.cpp.tmpl`            | `--template-dir`                          | gen_baseline_solution only      |
| `<op_type>.ncnn_kernel.cpp.tmpl`        | `--template-dir`                          | gen_baseline_solution only      |

A different backend points `--template-dir` at its own directory with the same 3-file naming
convention — it doesn't share `scripts/baseline_binding_templates/` with ncnn.

---

## Common bugs

| Symptom | Root cause | Fix |
|---------|------------|-----|
| `WARNING: No baseline traces produced` for all definitions | baseline solutions not in `bench-trace/` | Run `gen_baseline_solution.py` for all op types; rsync to remote |
| `FileNotFoundError: libncnn.a not found` on Graviton | ncnn not built | `cd ncnn && cmake -B build … && cmake --build build -j$(nproc) ncnn` |
| `undefined symbol: omp_get_num_threads` at dlopen | Missing `-fopenmp` in link_flags | Ensure solution JSON has `"link_flags": ["-fopenmp"]` |
| `IndexError`/garbled args reaching `armbench_entry_<op_type>` | binding.cpp's parameter count doesn't match how many real tensors the definition declares | Signature must be exactly `(primary_v, top_v, *rest_v, opt_v)` — no padding slots (see ABI section) |
| SIGSEGV in `layer.forward()` | `ncnn::Mat` dimensions in wrong order, or a field the constructor never initialized was left unset | ncnn uses `(w, h, c)` order; re-check every field against the real header (`Pooling_arm`/`LSTM_arm` especially — see Step 2 table) |
| Output tensor mismatched (wrong values) | Bulk-`memcpy`ing a whole Mat instead of row/channel-by-row/channel | Always iterate rows (`.row(i)`) or channels (`.channel(c)`) to skip `cstep` alignment padding |
| lstm output numerically wrong but same shape | PyTorch `I,F,G,O` gate order used as-is instead of permuting to ncnn's `I,F,O,G` | Swap chunks 2 and 3 when building `weight_xc_data`/`weight_hc_data`/`bias_c_data` |
| `create_pipeline` returns non-zero | Wrong layer params (e.g. kernel_w > W+2*pad), or an unset field read as garbage | Print the error; check every field was explicitly set (see Step 2) and pad/stride/kernel against the workload shape |
| `FileNotFoundError: Baseline kernel template not found` | Missing one of the 3 `.tmpl` files | Write it to `--template-dir` (default `scripts/baseline_binding_templates/`) |
| `_SkipDefinition` for every definition in a dir | Wrong `--tag`, or the dir mixes multiple backends' definitions (e.g. `gemm/` has both `ncnn`- and `llama.cpp`-tagged defs) | Check the definition's `tags` list; pass the matching `--tag` |
| `author='baseline-ncnn-arm'` but file under `reference-scalar/` | Path/content drift — TraceSet raises `ValueError` at load time | Ensure `--author`/`--dataset` match the output dir the script actually wrote to |
