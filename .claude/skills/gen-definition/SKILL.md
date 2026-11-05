---
name: gen-definition
description: Define a brand-new op type as a bench-trace Definition JSON with kgen; use when the user wants to benchmark an operator not yet in bench-trace/definitions/
---

## gen_definition

Design a Definition JSON for a new operator, then place it with `python -m kgen add definition`.

**This skill writes the structural schema only.** Workloads are added afterward with
`/gen-workload`; solutions with `/gen-candidate-solution` and `/gen-baseline-solution`.

The `reference` inside the definition is a correctness oracle, not a workload.

kgen commands used here:

| Command | Purpose |
|---|---|
| `kgen show <def> [--full]` | read a similar existing definition (axes, tensors, tags, constraints, reference with `--full`) before writing |
| `kgen add definition FILE [--replace]` | lint FILE and write `bench-trace/definitions/<op_type>/<name>.json` |
| `kgen discover plugin.py [--out F] [-- args]` | frameworks only: list the operators a model uses (see Step 0) |
| `kgen check <name> --static-only` | re-lint what is in the warehouse |

---

## What to extract from the user's message

- **Op name / description** - what it computes (enough to write a reference). Ask if ambiguous.
- **Axes** - which are fixed per kernel config (`const`) and which vary per workload (`var`).
- **Inputs / outputs** - names, shapes (lists of axis names), dtypes.
- **Constraints** - equations for derived axes.
- **Reference** - a NumPy/PyTorch `run(...)` that is correct for any valid inputs.

If any of these is underspecified, ask before writing.

---

## Definition JSON schema

The existing files are the ground truth: `kgen show <similar-definition> --full` first.

```json
{
  "name":        "<name>",
  "op_type":     "<op_type>",
  "description": "<human-readable description>",
  "tags":        ["status:active", "baseline-solution:<ncnn|llama.cpp|kleidiai>"],
  "axes": {
    "<axis>": { "type": "var" },
    "<axis>": { "type": "var", "parent": "<other_var_axis>" },
    "<axis>": { "type": "const", "value": <integer> }
  },
  "inputs":  { "<name>": { "shape": ["<axis>", ...], "dtype": "<dtype>" },
               "<scalar>": { "shape": null, "dtype": "<dtype>" } },
  "outputs": { "<name>": { "shape": ["<axis>", ...], "dtype": "<dtype>" } },
  "constraints": ["<derived_axis> == <expression using other axes>"],
  "reference": "<Python source string - a def run(...): function>"
}
```

**`axes`**: `var` changes between workloads (batch, spatial dims, sequence length); `const` is fixed
for the definition (kernel size, stride, N and K of a gemm). Only `var` axes appear in a
workload's `axes`. `parent` on a `var` axis groups it under another `var` axis of the same row
(e.g. `H_out` derived from `H`).

**`inputs` shapes**: every axis named must be in `axes`. `"shape": null` is a scalar passed as a
fixed value in each workload.

**`constraints`**: only for derived `var` axes; Python integer floor division `//` for conv output
sizes.

**`reference`**: exactly one function `run`, one positional argument per input in declaration order.
Array inputs arrive as NumPy arrays, scalars as Python numbers; return the output (or a tuple for
several). Minimal PyTorch/NumPy, no comments or prints, const axis values inlined as integer
literals (the string cannot see axis names).

### Naming and path

- `name`: lowercase with underscores, encoding the const axes that distinguish variants
  (`rms_norm_d768`, `gemm_q4_K_n2048_k2048`).
- `op_type`: usually the first component of the name.
- `kgen add` puts the file at `bench-trace/definitions/<op_type>/<name>.json` (simd-loop shares one
  directory). It refuses an existing target without `--replace` and writes nothing when lint fails.

---

## Workflow

### Step 0: Find the operators

Write a `discover` plugin if you think you need it. Then run it:

```bash
python -m kgen discover plugin.py [--out inventory.json] [-- plugin args]
```

The plugin defines `discover(argv) -> dict` (read a GGUF header, trace a torchvision model, parse an ONNX graph, list a library's micro-kernels or loop sources ...). kgen only loads it and checks the result is a JSON object; what the dict holds is yours to decide. `plugins/llamacpp/inventory.py` is an example. Both a framework and a library may use a plugin; the criteria differ. Have the plugin apply them and report what it dropped and why.

**Framework (operators come from a model)**; record the model as a `model:` tag:

- **Which operators are candidates**:
  Graph Execution Only: Covers only the real executed graph (excluding unused layers like MTP heads, or lookup-only tensors like untied embeddings).
  bench-trace Compatibility: Expressed as a definition supported by bench-trace (supported dtypes/layouts, explicitly listing unsupported elements under "not covered").
  No Redundancy: Excludes anything already defined in bench-trace.

- **Which candidates are kept**:
  Identify Execution Candidates (role, type, shape):
    Filter for unique operator candidates defined by their role, data type, and tensor shape to prevent redundant benchmarking of identical kernels.Skip operators too small to time reliably to eliminate measurement noise.
  Rank Candidates by Time Share (Not Count):
    Rank operators using estimated execution time rather than raw layer or call counts.Operators with higher time share are more critical to benchmark, as they contribute more to overall execution time.
    Feel free to customize your ranking metric based on the model/kernel's characteristics, for example: For memory-bound decode, estimate time share based on weight bytes accessed per token.For MoE experts, scale memory/compute weight by the active routing fraction per token (e.g., top-$k$ active experts).
  Apply Pareto Tail Cut (90% Cumulative Share per Group):
    Group candidates by (model, quantization type).
    Within each group, rank candidates by share, calculate cumulative share relative to the group total, and retain operators until the running total first reaches or crosses 90%.
    Drop all remaining long-tail operators beyond the crossing point.
  Provide Audit Trail & Context:
    List all dropped operators explicitly for transparency.
    State explicitly whether the time share is estimated or measured.

**Library (operators come from a kernel catalogue)**:

- **Distinct**: one candidate per (operation, dtype/layout, ISA feature); other tile sizes or parameter values of the same kernel are workloads or const axes, not new operators.
- **Runnable**: its ISA (SVE, SVE2, SME2 ...) exists on a target machine; list the rest as "needs <ISA>" instead of dropping them.
- **Has a reference**: its behavior fits a short `run(...)`, or the library ships one.
- **Big enough to measure**: skip kernels whose smallest sensible input is too small to time.
- **Not already defined**: mark candidates whose definition is already in `bench-trace`.
- **Has a baseline**: the library's own implementation, or the next best one (the autovectorized loop for simd-loop); without one, mark it as not evaluable.

If the user names the operator directly, no plugin is needed.

### Conventions

**`model:` tag**: lowercase, hyphens (`model:olmoe-1b-7b`, `model:resnet50`); omit for definitions
not tied to a model.

**`baseline-solution:<framework>` tag**: every definition except simd-loop carries exactly one,
naming the library whose kernel is the speedup baseline. **Always ask the user which framework;
do not infer it from the `model:` tag or dtype** (they correlate today but are not a rule). It
records intent only - check `bench-trace/solutions/` for what actually exists. Omit it for
simd-loop.

**dtype variants** are separate definitions with the scheme in the name:

| name pattern | inputs | scheme |
|---|---|---|
| `gemm_fp32_n{N}_k{K}` | float32 | plain |
| `gemm_w8a8ch_n{N}_k{K}` | int8 + scales | ncnn per-channel: `A [M,K]` int8 + scalar `input_scale`; `B [N,K]` int8 + `weight_scales [N]`; output int8 |
| ggml block types (`gemm_q4_K_n{N}_k{K}`, `gemm_q8_0_...`) | bf16 `A [M,K]`, `B` as packed ggml blocks `uint8 [N, K_bytes]` | llama.cpp's own block layout; K must be a multiple of the block size |

Keep the quantized weights in the framework's real storage layout; do not re-encode as separate
value and scale tensors unless the framework does. MoE routing weights stay float32. For MHA use
`[M, n_heads, head_dim]`; baseline wrappers handle transposes. Quantized definitions are
usually tagged `correctness:sqnr` (judged by SQNR against the baseline, see `/gen-workload`
step 5).

### Step 1: Survey

`ls bench-trace/definitions/` and `kgen show` on one or two definitions in the closest family.

### Step 2: Elicit the design

Settle with the user: the math in one sentence; const vs var; axis names (same casing as existing
definitions: `C_in`, `H`, `Kh`); input/output names, shapes, dtypes; constraints; scalar inputs;
and the baseline framework.

### Step 3: Present the plan

Show the complete definition (axes, tensors, constraints, reference, tags) and the output path,
and wait for explicit confirmation before writing.

### Step 4: Write and add

Write the JSON to the scratchpad, then:

```bash
python -m kgen add definition /path/to/<name>.json
```

Each problem comes back as one sentence naming the field; fix the file and run it again. For a
batch of related definitions, write one throwaway snippet in the scratchpad that emits the files
and `add` each of them.

### Step 5: Verify and hand over

`python -m kgen check <name> --static-only` (a definition without workloads reports that its
workload file does not exist - expected until the next step), then tell the user:

```
Definition written to bench-trace/definitions/<op_type>/<name>.json.
Add workloads next: /gen-workload - add workloads to <name>
```

---

## Common problems

| Symptom | Cause | Fix |
|---------|-------|-----|
| `axes.X: ...` schema error | const axis without `value`, wrong `type` | Re-check each axis entry |
| `axes.X.parent: 'Y' is not declared` | `parent` names a missing axis | Declare it or drop `parent` |
| `inputs and outputs share names` | Same tensor name on both sides | Rename |
| `reference` fails with `NameError` | An axis name used symbolically | Inline the const value as an integer |
| Output shape mismatch in `check` | Wrong constraint (off by one, wrong formula) | Re-derive by hand with a small example |
| Guessed `baseline-solution:` from the model/dtype | Correlation is not a rule | Ask the user |
