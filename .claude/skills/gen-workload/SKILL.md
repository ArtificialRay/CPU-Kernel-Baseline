---
name: gen-workload
description: Add or refresh the workloads of a definition with kgen (pick shapes from a real model run, real activations, calibration); use when a definition needs workloads or its workloads must be regenerated
---

## gen_workload

Give a definition its workloads (concrete var-axis values plus how each input is produced) with
`python -m kgen`. kgen holds only what is the same for every framework; anything specific to one
model, framework or operator class (which prompts to feed, which layer is "the operator") is a
small file **you write at run time** and pass in. There are no generator scripts to run.

| Command | What it does |
|---|---|
| `kgen show <def>` | axes, tensors, tags, the workload points and solutions already there |
| `kgen add workload FILE --definition D [--replace] [--max-count N --batch-key AXIS]` | lint FILE and write `bench-trace/workloads/<op>/<D>.jsonl` |
| `kgen collect pytorch --loader load.py ... --out-dir DIR` | run a PyTorch model, write the shapes its layers saw as candidate pools `DIR/<def>.jsonl` |
| `kgen capture pytorch --loader load.py --source TEXT ...` | replace an input of existing workloads with real activations from the same run |
| `kgen calibrate --author A --baseline-author B ...` | (aarch64 only) record the baseline's SQNR per workload for `correctness:sqnr` definitions |
| `kgen check <def> [--static-only]` | lint everything; the execution legs run kernels and need aarch64 |

`--root` (before the subcommand for `add`: `add --root R --replace workload ...`) picks another
warehouse; the default is `bench-trace`.

---

## Workflow

### 1. Read what exists

`kgen show <def>`. Note the var axes (only these go in a workload's `axes`; const axes live in the
definition), the input names and dtypes, and whether workloads already exist. Ask the user which
definition(s) if it was not stated.

### 2. Decide where the shapes come from

| Situation | Source |
|---|---|
| The operator comes from a PyTorch-runnable model (LLM linear/norm/MoE, CNN conv/pool) | `collect pytorch`, then `add workload --max-count` |
| A few specific points, or no model to run (library kernels such as simd-loop, kleidiai) | write the JSONL by hand, `add workload` without `--max-count` |

Present the planned points (a table with the rationale) and wait for the user's confirmation
before writing when you are choosing them by hand.

### 3a. From a model run: collect, then pick

Write one loader file in the scratchpad. It defines two functions (the contract, with a full
Linear example, is the docstring of `kgen/capture.py`):

```python
from kgen.capture import Adapter

def load():       # -> (model in eval mode, batches)   what the model is fed decides the shape distribution
def adapters():   # -> [Adapter(kind, match, params, named, axes, slice_axis), ...]  which layers are the operator
```

A layer feeds a definition when the definition's const axes agree with `params(layer)` (at least
two of them, or all if it has fewer) and the definition's input has exactly the adapter's `axes`.
Write one adapter per operator class you need (Linear, Conv2d, RMSNorm, an MoE block ...); kgen
knows none of them. For an LLM, prompt lengths are the distribution: feed real prompts (ask
before downloading a dataset), include a single-token batch so M=1 exists, and truncate to the
runtime's batch limit (512 for llama.cpp) so the range stays comparable.

```bash
python -m kgen collect pytorch --loader load.py --definitions D1 D2 --out-dir pools/
python -m kgen add workload pools/D1.jsonl --definition D1 --max-count 20 --batch-key M          # new file
python -m kgen add --replace workload pools/D1.jsonl --definition D1 --max-count 20 --batch-key M  # refresh
```

- The pool rows may be just `{"axes": {...}}`. Missing `inputs` are taken from the definition's
  existing workloads (same axes: kept as they are, real tensors included; new axes: the first
  row's inputs with `tensor` turned into `random`, so `bytes` layouts and scalars carry over). A
  definition with no workloads yet needs full rows with `inputs`.
- A pool may be handed to another definition with the same var axes (`--definition` names the
  target): prompt lengths do not depend on the model.
- `--max-count` filters in two layers: dedup (`--max-dups`, default 2 copies of identical axes),
  then greedy farthest-point diversity over `--batch-key` (default: the first axis). When there
  are more distinct batch values than `--max-count` the seeds are spread over the sorted range,
  smallest and largest included. Without `--max-count` every row is written as given.
- `add` refuses to overwrite without `--replace`; with it, a row whose axes already exist keeps
  its uuid. Nothing is written when lint finds a problem.
- `collect` prints `no layer of the model matches this definition` for definitions no adapter
  fits; fix the adapter's `params`/`axes` or the loader rather than hand-writing around it.

Coverage to check afterwards (`kgen show`): a var-axis value that is not a multiple of common SIMD
widths (3, 11 ...), odd spatial sizes for convolutions (about 15-20% of the points), the decode
point `M=1`, and for attention the point with `M=1` and the largest `S`.

### 3b. By hand

Write the JSONL: one object per line, no uuid needed (a missing one is assigned from the
definition name and axes; a given one is kept):

```json
{"axes": {"M": 64}, "inputs": {"A": {"type": "random"}, "B": {"type": "bytes", "layout": "ggml_q8_0"}}, "tags": {"from": "manual"}}
```

`axes` holds exactly the var axes that appear in the input shapes. Input types: `random`
(for tensors), `scalar` (`{"type": "scalar", "value": v}`, for `shape: null` inputs; a
quantization scale is a bounded random value such as `uniform(0.005, 0.05)` seeded by the
definition name, not a shared magic constant), `tensor` (a `.npy` under the warehouse, written by
capture), `bytes` (with a `layout`, generated at run time). Then `kgen add workload FILE
--definition D`.

### 4. Real activations (quantized operators)

Random activations hide numerical shortcuts that real post-norm hidden states expose, so a
quantized definition's A input should come from the model:

```bash
python -m kgen capture pytorch --loader load.py --source "<model / data / chunk>" --definitions D1 D2
```

The same loader and adapters as collect. Each workload with a fitting sample becomes a `tensor`
input (uuid unchanged, tagged `inputs: real-activations`); the others stay as they were and are
counted in the output. Pass `--definitions` or a narrow `--tag`: a broad tag also selects
definitions you did not mean. Rows with large M may not be capturable (an MoE expert sees about
T/8 rows per call); they stay random, and `kgen show` tells which workloads are real.

### 5. Calibrate (definitions tagged `correctness:sqnr`, aarch64)

```bash
python -m kgen calibrate --author reference-scalar --baseline-author <baseline author> --definitions D1 D2
```

It records `baseline_sqnr_db` on each workload: what the baseline scores on exactly those inputs.
Any change to a workload's inputs (capture, `--replace`, a regenerated pool) makes the recorded
floors stale; recalibrate before trusting `check`. Authors per dataset are in
`config/kernel_contracts.yaml`.

### 6. Verify

```bash
python -m kgen check <D> --static-only      # anywhere: schema, axes, tensor files exist with the right shape/dtype
python -m kgen check <D> --num-selected-workloads 5     # on aarch64: runs reference, reference-scalar, baseline
```

---

## Reference

### Var axes by op family

| Op family | Var axes | Notes |
|-----------|----------|-------|
| `conv2d` | `N, C_in, H, W` | N=1 only (ncnn adapter) |
| `conv2d_depthwise` | `N, C, H, W` | C_in = C_out = C; N=1 only |
| `conv1d` | `C_in, W` | no N axis |
| `pooling` | `N, C, H, W` | all four vary with input resolution |
| `simd-loop` | `N` | element count |
| `gemm` | `M` | K, N are const |
| `rms_norm` | `M` | D const |
| `mha` | `M, S` | M = query len, S = KV len |
| `moe` | `n_tokens` | total tokens entering the MoE block (not per-expert counts) |

### Common problems

| Symptom | Cause | Fix |
|---------|-------|-----|
| `axes missing ['C_in']` | A var axis that appears in an input shape is absent | Pass every var axis of the definition |
| `axes.K is const in the definition, leave it out` | A const axis was put in `axes` | Remove it |
| `X already exists; pass --replace` | Target workload file exists | Add `--replace` (before `workload`) if overwriting is intended |
| `row N has no inputs, and the definition has no workloads` | Axes-only pool for a definition with no workloads yet | Write full rows with `inputs` |
| `no layer of the model matches this definition` | No adapter's `params`/`axes` fits the definition | Fix the adapter or loader |
| `an adapter failed on layer ...` | Exception inside the loader's `match/params/named` | Fix the adapter; the message names the layer |
| `inputs.A file has dtype ...` | `.npy` dtype cannot be loaded as the declared one | bf16/fp16 are stored as 2-byte patterns or float32; others must cast safely |
| `check` fails on `baseline_sqnr_db` after editing workloads | Floors measured on the old inputs | Re-run `calibrate` |
| ncnn `N=1 only` error | Workload has N>1 for an ncnn-backed definition | Use N=1 |
