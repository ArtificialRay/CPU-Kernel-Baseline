---
name: gen-baseline-solution
description: Generate or regenerate baseline-ncnn-arm solution JSONs for an op type; use when adding a new op type baseline or after definitions are regenerated
---

## gen_baseline_solution

Generate `baseline-ncnn-arm` solution JSONs for a given op type. Each solution embeds
all three source files directly: the harness (`.h`, `.cpp`) is shared with
reference-scalar, and `kernel.cpp` delegates to `ncnn::*_arm` layers linked from
`libncnn.a`. `NcnnBuilder` compiles these against the full ncnn static lib.

Script: `scripts/gen_baseline_solution.py`

---

## What to extract from the user's message

Skills have no formal parameters — extract context from the user's natural language request:

- **Op type** — which operator to generate baselines for (e.g. `conv2d`, `conv1d`,
  `conv2d_depthwise`, `deconv2d`, `deconv2d_depthwise`). If not mentioned, ask before proceeding.
- **Scope** — a specific definition name to target, or all definitions in the op type (default).
- **Mode** — generating from scratch (new op type) or regenerating after a definitions refresh.

Valid invocations look like:
```
Generate baseline-ncnn-arm solutions for all 5 op types
Regenerate deconv2d baselines after the definitions were updated
Add baseline solutions for the new conv1d op type
```

---

## What a baseline solution JSON looks like

Every baseline solution is a single JSON file at:
```
bench-trace/solutions/ncnn/baseline-ncnn-arm/<op_type>/<def_name>.json
```

Example for `conv2d`:
```json
{
  "name":        "baseline-ncnn-arm_<def_name>",
  "definition":  "<def_name>",
  "dataset":     "ncnn",
  "author":      "baseline-ncnn-arm",
  "description": "ncnn::*_arm baseline for <def_name>. Same harness as reference-scalar; kernel.cpp delegates to libncnn.a. Timing baseline for speedup computation.",
  "spec": {
    "language":        "cpp",
    "target_hardware": ["graviton3", "aarch64-sve", "graviton4", "aarch64-sve2"],
    "entry_point":     "<op_type>.cpp::armbench_entry_<op_type>",
    "dependencies":    [],
    "isa_features":    [],
    "compile_flags":   ["-O3", "-std=c++17"],
    "link_flags":      ["-fopenmp"]
  },
  "sources": [
    { "path": "<op_type>.h",   "content": "// constexpr params baked from definition..." },
    { "path": "<op_type>.cpp", "content": "// binding harness: armbench_entry_<op_type> → inner_<op_type>..." },
    { "path": "kernel.cpp",    "content": "// calls ncnn::*_arm layer (links against libncnn.a)..." }
  ]
}
```

Key constraints:
- `dataset` must be `"ncnn"` — `NcnnBuilder` is activated by `is_baseline=True` AND `dataset=="ncnn"`
- `author` must be `"baseline-ncnn-arm"` exactly — `Benchmark._is_baseline()` checks `solution.author == config.baseline_author`
- `entry_point` must be `"<op_type>.cpp::armbench_entry_<op_type>"`, same as reference-scalar
- `compile_flags` must include `-std=c++17` — ncnn headers require C++17
- `link_flags` must include `-fopenmp` — `libncnn.a` references OpenMP symbols

---

## Workflow: regenerate existing op type

If the op type is already registered and templates already exist:

```bash
# All definitions for one op type
python -m scripts.gen_baseline_solution --op-type <op_type>

# Single definition
python -m scripts.gen_baseline_solution --op-type <op_type> --definition <def_name>

# All 5 op types at once
for op in conv2d conv1d conv2d_depthwise deconv2d deconv2d_depthwise; do
    python -m scripts.gen_baseline_solution --op-type $op
done

# Sync to Graviton (bench-trace/ is gitignored — must rsync manually)
rsync -az --delete \
    bench-trace/solutions/ncnn/baseline-ncnn-arm/ \
    ubuntu@<host>:arm-bench/bench-trace/solutions/ncnn/baseline-ncnn-arm/

# Collect baselines on remote to verify they compile and run correctly
ssh ubuntu@<host> "cd arm-bench && python3 -m bench.cli collect-baselines --baseline-author baseline-ncnn-arm 2>&1"
```

---

## Workflow: add a new op type

### Step 1: Ensure harness templates exist

The baseline script reads harness templates from `scripts/candidate_binding_templates/`:
- `<op_type>.h.tmpl` — constexpr header (shared with reference-scalar)
- `<op_type>.cpp.tmpl` — binding harness (shared with reference-scalar)

These must exist before writing the kernel template. If adding a completely new op type,
follow the `gen-candidate-solution` skill to create them first.

---

### Step 2: Understand the ncnn layer API for this op type

Identify which `ncnn::*_arm` class to use and how to construct it:

| Op type              | ncnn class                    | Header                            |
|----------------------|-------------------------------|-----------------------------------|
| conv2d               | `ncnn::Convolution_arm`       | `convolution_arm.h`               |
| conv1d               | `ncnn::Convolution_arm`       | `convolution_arm.h` (H=1 trick)   |
| conv2d_depthwise     | `ncnn::ConvolutionDepthWise_arm` | `convolutiondepthwise_arm.h`   |
| deconv2d             | `ncnn::Deconvolution_arm`     | `deconvolution_arm.h`             |
| deconv2d_depthwise   | `ncnn::DeconvolutionDepthWise_arm` | `deconvolutiondepthwise_arm.h` |

Key settings to apply on every layer object before `create_pipeline()`:
- `opt.use_packing_layout = false` — avoids weight repacking overhead per call; output stays NCHW float32
- `opt.num_threads = 1` — single-threaded to match the timing harness pinned-core mode
- `bias_term = 1` if the op takes a bias input; set `bias_data` from the caller's pointer
- `dynamic_weight = 0` — weight is constant, set `weight_data` before `create_pipeline()`

---

### Step 3: Write the kernel template

Location: `scripts/baseline_binding_templates/<op_type>.ncnn_kernel.cpp.tmpl`

The template uses the same `{{placeholder}}` substitution as the harness templates;
`gen_baseline_solution.py` fills these from the definition's const axes via `op_config.json`.

**Pattern for a batched op** (conv2d, deconv2d):
```cpp
#include "<op_type>.h"
#include "<ncnn_layer_header>"
#include "mat.h"
#include "option.h"
#include <cstring>
using namespace <op_type>_def;

void inner_<op_type>(const float* input, float* output, const float* weight,
                     int N, int C_in, int H, int W, int H_out, int W_out) {
    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = false;

    const int weight_size = Cout * C_in * Kh * Kw;
    ncnn::Mat weight_mat(weight_size, (void*)weight, (size_t)4u);

    ncnn::SomeLayer_arm layer;
    layer.num_output = Cout;  layer.kernel_w = Kw;  layer.kernel_h = Kh;
    layer.stride_w   = Sw;    layer.stride_h = Sh;
    layer.dilation_w = Dw;    layer.dilation_h = Dh;
    layer.pad_left   = pad_left;  layer.pad_right  = pad_left;
    layer.pad_top    = pad_top;   layer.pad_bottom = pad_top;
    layer.pad_value = 0.f;
    layer.bias_term = 0;  layer.weight_data_size = weight_size;
    layer.int8_scale_term = 0;
    layer.activation_type = activation_type;  layer.activation_params = ncnn::Mat();
    layer.dynamic_weight = 0;
    layer.weight_data = weight_mat;
    if (layer.create_pipeline(opt) != 0) return;

    for (int n = 0; n < N; ++n) {
        ncnn::Mat bottom(W, H, C_in, (void*)(input + (long)n * C_in * H * W), (size_t)4u);
        ncnn::Mat top;
        if (layer.forward(bottom, top, opt) != 0) return;
        for (int c = 0; c < Cout; ++c)
            std::memcpy(output + (long)n * Cout * H_out * W_out + (long)c * H_out * W_out,
                        (const float*)top.channel(c), H_out * W_out * sizeof(float));
    }
}
```

**`ncnn::Mat` wrapping rules:**
- `ncnn::Mat(W, H, C, (void*)ptr, (size_t)4u)` — wraps external float* without copying; dimensions are (W, H, C) in ncnn's column-major convention
- `top.channel(c)` — returns a pointer to channel c; must use `memcpy` to copy `H_out*W_out` floats because channels may have alignment padding
- For conv1d: treat the sequence as a 1D conv by setting `kernel_h=1`, `stride_h=1`, `pad_top=0` and wrapping input as `ncnn::Mat(W, 1, C_in, ...)`
- For depthwise: set `group = C` on the layer object; `num_output = C`

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

| File type                         | Directory                                | Script that reads it        |
|-----------------------------------|------------------------------------------|-----------------------------|
| `<op_type>.h.tmpl`                | `scripts/candidate_binding_templates/`   | both gen scripts            |
| `<op_type>.cpp.tmpl`              | `scripts/candidate_binding_templates/`   | both gen scripts            |
| `<op_type>.kernel.cpp.tmpl`       | `scripts/candidate_binding_templates/`   | gen_candidate_solution only |
| `<op_type>.ncnn_kernel.cpp.tmpl`  | `scripts/baseline_binding_templates/`    | gen_baseline_solution only  |
| `op_config.json`                  | `scripts/candidate_binding_templates/`   | both gen scripts            |

---

## Common bugs

| Symptom | Root cause | Fix |
|---------|------------|-----|
| `WARNING: No baseline traces produced` for all definitions | `baseline-ncnn-arm` solutions not in `bench-trace/` | Run `gen_baseline_solution.py` for all op types; rsync to remote |
| `FileNotFoundError: libncnn.a not found` on Graviton | ncnn not built | `cd ncnn && cmake -B build … && cmake --build build -j$(nproc) ncnn` |
| `undefined symbol: omp_get_num_threads` at dlopen | Missing `-fopenmp` in link_flags | Ensure solution JSON has `"link_flags": ["-fopenmp"]` |
| SIGSEGV in `layer.forward()` | `ncnn::Mat` dimensions in wrong order | ncnn uses `(W, H, C)` order — double check args to `ncnn::Mat(...)` |
| Output tensor mismatched (wrong values) | Using `top.data` instead of `top.channel(c)` | Always iterate channels with `top.channel(c)` + memcpy to skip alignment padding |
| `create_pipeline` returns non-zero | Wrong layer params (e.g. kernel_w > W+2*pad) | Print the error; check pad/stride/kernel against the workload shape |
| `FileNotFoundError: Baseline kernel template not found` | Missing `<op_type>.ncnn_kernel.cpp.tmpl` | Write it to `scripts/baseline_binding_templates/` |
| `author='baseline-ncnn-arm'` but file under `reference-scalar/` | Path/content drift — TraceSet raises `ValueError` at load time | Ensure output dir in the script is `solutions/ncnn/baseline-ncnn-arm/<op_type>/` |
