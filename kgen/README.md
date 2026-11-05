# kgen

`python -m kgen` is a command-line tool for managing and adding kernels to `bench-trace/`. Designed to be driven by either an AI agent or a user, it handles linting, placing files, checking implementations, and selecting workloads uniformly across frameworks.

---

## Architecture Overview

A kernel consists of four standard files across three main categories:

| File Type | Purpose | File Location |
| --- | --- | --- |
| **Definition** | Declares axes, tensors, constraints, and a Python `reference` | `bench-trace/definitions/<op_type>/<name>.json` |
| **Workloads** | Specifies concrete var-axis values and input generation rules | `bench-trace/workloads/<op_type>/<name>.jsonl` |
| **Solutions** | Houses the reference-scalar starting point and the library baseline | `bench-trace/solutions/<dataset>/<author>/<op_type>/<name>.json` |

*(Note: SIMD-loop definitions share one directory and nest solutions one level deeper.)*

### Core Design Principles

* **Framework-Agnostic Core:** `kgen` handles only common infrastructure (linting, validation, file placement, and workload management). It contains no generator scripts and generates no kernels directly.
* **Extensible Plugins & Loaders:** Framework-, model-, or operator-specific logic is supplied via external helper files provided by the user or coding agent.

---

## Command Reference

### Summary of Commands

| Command | Primary Action |
| --- | --- |
| `add` | Lints and writes a definition, workload, or solution to `bench-trace/`. |
| `show` | Displays axes, tensors, tags, workload points, and solutions of a definition. |
| `check` | Runs static and execution verification tests across all four validation legs. |
| `discover` | Runs a custom plugin script to extract and list a framework's operators. |
| `collect` | Hooks into a PyTorch model to capture real layer execution shapes into a pool. |
| `capture` | Replaces a workload's synthetic input with real saved activation tensors. |
| `calibrate` | Records baseline SQNR scores for quantized definitions into workload tags. |
| `autovec` | Benchmarks a baseline against compiler auto-vectorized scalar implementations. |

---

### Key Workflows & Detailed Usage

#### 1. Adding Files (`add`)

Validates a target file and writes it to the repository. If errors exist, it outputs a sentence describing each issue (field, expected, found) and cancels the write.

* **Overwrite Flag:** Use `--replace` if the target file already exists.
* **Root Flag:** Specify `--root <DIR>` (must precede the subcommand, e.g., `add --root custom_path workload ...`).

**Workload Structure:**
Each line in a `.jsonl` workload file follows this format:

```json
{"axes": {...}, "inputs": {...}, "tags": {...}}

```

* `axes`: Must strictly match the variable axes declared in the definition.
* `inputs`: Supports `random`, `scalar` (`{"type": "scalar", "value": v}`), `tensor` (pointing to a `.npy` file), or `bytes` (generated at runtime with a specific `layout`).
* `uuid`: Optional. Reuses an existing UUID if matching axes are found; otherwise generates a new one.

**Selecting from a Pool:**

```bash
python -m kgen add workload --max-count N --batch-key AXIS [--max-dups 2]

```

Filters candidates from a pool (lines containing `{"axes": {...}}`) by picking $N$ rows using farthest-point diversity along the specified batch axis. Missing inputs inherit from the definition's existing workloads.

---

#### 2. Verification & Linting (`check`)

Runs a multi-stage validation check on definitions and implementations:

1. **Static:** Verifies that definition, workload, and solution files pass standard lint rules (`--static-only` runs this phase exclusively).
2. **Reference:** Ensures the Python `reference` executes on every workload and matches declared shapes.
3. **Reference-Scalar:** Verifies that the scalar C/C++ solution builds, runs, and matches output with the reference.
4. **Baseline:** Verifies that the real framework baseline library builds, runs, and matches output.

> **Hardware Requirement:** Execution legs (2–4) build and run compiled kernels, requiring ARM AArch64 hardware with supported instruction sets (controlled via `--isa` and `--num-selected-workloads N`).

---

#### 3. Execution Tracing (`collect` & `capture`)

Hooks into a PyTorch model execution using a standardized loader script (`load.py`).

```python
# Example load.py
from kgen.capture import Adapter

def load():
    return model, batches  # Model in eval mode; batches for model(...)

def adapters():
    return [
        Adapter(
            kind="linear",
            match=lambda layer: type(layer).__name__ == "Linear",
            params=lambda layer: {"K": layer.in_features, "N": layer.out_features},
            named=lambda layer, x: (x.reshape(-1, x.shape[-1]), ("M", "K")),
            axes=("M", "K"),
            slice_axis="M"
        )
    ]

```

* **`collect pytorch`**: Sweeps model layers and writes unique observed shape configurations to a pool (`DIR/<definition>.jsonl`).
```bash
python -m kgen collect pytorch --loader load.py (--definitions D... | --tag T) --out-dir DIR

```


* **`capture pytorch`**: Saves actual tensor activations as `.npy` files and updates workloads with `tensor` inputs tagged `inputs: real-activations`.
```bash
python -m kgen capture pytorch --loader load.py --source TEXT (--definitions D... | --tag T)

```



---

#### 4. Baseline Calibration (`calibrate`)

Measures the Signal-to-Quantization-Noise Ratio (SQNR) of quantized kernels (`correctness:sqnr`) by running `--author` against `--baseline-author`.

* Stores the baseline accuracy floor (`baseline_sqnr_db`) directly into workload tags.
* Evaluators will flag any implementation falling more than `EvalConfig.sqnr_margin_db` (default 1 dB) below this baseline.
* **Note:** Must be re-run whenever activation inputs are updated via `capture`.

---

#### 5. Baseline Sanity Check (`autovec`)

Verifies that a baseline implementation is genuinely optimized rather than a naive loop.

* Rebuilds the scalar starting point using target compiler flags (`-march`) to enable auto-vectorization.
* Flags the baseline if its geometric-mean execution time is more than `--max-slowdown` (default 2x) slower than the auto-vectorized scalar build.
* Must be executed directly on the target tier's native hardware.

---

#### 6. Custom Operator Discovery (`discover`)

Imports and executes a custom Python script defining `discover(argv) -> dict` to inspect framework operators or shapes.

```bash
python -m kgen discover PLUGIN [--out output.json] [-- plugin_args...]

```

---

## Repository Structure & Ecosystem

* `plugins/`: Archived discovery plugins used to extract SIMD loops and llama.cpp GEMM metadata.
* `.claude/skills/`: Agent workflows for generating definitions, workloads, and solutions:
* `gen-definition`
* `gen-workload`
* `gen-candidate-solution`
* `gen-baseline-solution`