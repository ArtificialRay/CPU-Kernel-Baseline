# CPU-Kernel-Baseline
CPU Kernels/Operators for vision model / LLM extracted from 5 codebases, serving as baseline for the project

## Sync local codebase with remote instance

**Add new files:**
```bash
./arm-bench/sync_remote.sh
```

**Force mirror:**
```bash
./arm-bench/sync_remote.sh --mirror
```

**Change another instance via IP address:**
```bash
HOST=1.2.3.4 ./arm-bench/sync_remote.sh
```

---

## Prerequisites

**Install Python dependencies:**
```bash
pip install litellm python-dotenv
```

## Running the benchmark

All evaluation is driven by a single entry point:

```bash
python eval/run_benchmark.py --problem <op_type> --dataset <dataset> --model <model>
```

The benchmark script will prepare all relevant dependencies at remote instance.

---

### Usage examples

**Run a single op type (ncnn dataset, SVE2 / Graviton4 by default):**
```bash
python eval/run_benchmark.py --problem conv2d --dataset ncnn --model anthropic/claude-opus-4-8
```

**Run all definitions for a dataset:**
```bash
python eval/run_benchmark.py --all --dataset ncnn --model anthropic/claude-opus-4-8
```

**Provision a fresh instance, run, then tear it down automatically:**
```bash
python eval/run_benchmark.py --all --dataset ncnn --model anthropic/claude-opus-4-8 \
    --provision --teardown
```

**Override ISA (e.g. Graviton3 SVE):**
```bash
python eval/run_benchmark.py --all --dataset simd-loop --model anthropic/claude-opus-4-8 \
    --isa sve
```

**Run simd-loop dataset:**
```bash
python eval/run_benchmark.py --problem loop_001 --dataset simd-loop --model anthropic/claude-opus-4-8
```

---

### All options

| Flag | Default | Description |
|------|---------|-------------|
| `--problem <name>` | — | Definition name or op_type prefix (e.g. `conv2d`) |
| `--all` | — | Run all definitions for the dataset (mutually exclusive with `--problem`) |
| `--dataset` | `ncnn` | Dataset to benchmark: `ncnn` or `simd-loop` |
| `--model` | (required) | LiteLLM model string, e.g. `anthropic/claude-opus-4-8` |
| `--isa` | `sve2` | ISA target: `neon`, `sve`, `sve2`, `sme2` |
| `--provision` | off | Provision a new instance even if one is already configured |
| `--teardown` | off | Destroy the instance after evaluation |
| `--max-turns` | `20` | Max agent turns per definition |
| `--quiet` | off | Suppress per-turn output |
| `--no-save` | off | Don't save results to `results/` |
| `--save-trace` | off | Save full `version_history` to `traces/` |
| `--skip-baselines` | off | Skip lazy baseline collection (use if baselines are already present) |

### Instance types

| ISA | Instance | Notes |
|-----|----------|-------|
| `sve` | `c7g.large` | Graviton3, Neoverse V1, 256-bit SVE |
| `sve2` | `c8g.large` | Graviton4, Neoverse V2, 128-bit SVE2 (default) |

---

### Teardown

```bash
python eval/provision.py --teardown
```

Or pass `--teardown` directly to `run_benchmark.py` to destroy automatically after evaluation.
