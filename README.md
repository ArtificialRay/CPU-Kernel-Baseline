# CPU-Kernel-Baseline

Evaluates LLMs on writing optimized AArch64 SIMD kernels for ncnn / llama.cpp /
synthetic simd-loop benchmarks. 

FP32 Kernels available:
| Kernel Name | Type | Source |
|---|---|---|
| RMSNorm | Memory Bound | llama.cpp, ncnn |
| Conv2D Depthwise | Memory Bound | ncnn |
| Pooling (Reduction) | Memory Bound | ncnn |
| GEMM (M=1, decode) | Memory Bound | llama.cpp, ncnn |
| Conv2D | Compute Bound | ncnn |
| GEMM (M≥32, prefill) | Compute Bound | llama.cpp, ncnn |
| MHA | Fused | llama.cpp|
| LSTM | Fused | ncnn |

INT8 Kernels available:
| Kernel Name | Type | Source |
|---|---|---|
| Conv2D | Compute Bound | ncnn |
| GEMM | Memory Bound/Compute Bound | ncnn,llama.cpp |
| MoE | Fused | llama.cpp |

Kernel definitions are extracted from real model architecture: qwen1.5-moe-a2.7b, olmoe-1b-7b, resnet50, mobilenetv3-large, deepspeech2

---

## Prerequisites

```bash
pip install -r requirements.txt
```

Provisioning and remote runs need an AWS account with Terraform configured
(`terraform/`) and an SSH key. See `eval/eval_config.json.example`.

## Two ways to run an agent against this benchmark

- **Own harness** — this repo's own litellm agent loop (`eval/run_benchmark.py`)
  drives the session end-to-end: provisions a Graviton instance, starts
  `mcp_app/server.py` on it, and runs a self-contained tool-call loop against
  it. No external agent harness needed.
- **MCP server for an external harness** — start `mcp_app/server.py` directly
  (or via `skills/launch/`) and point an external agent harness (nanobot,
  Claude Code, ...) at it. This repo never drives the model in this mode; the
  external harness does.

Both modes share the same `compile`/`evaluate`/`disassemble`/`submit` tool
surface and the same local `bench/` library underneath — see
[CLAUDE.md](CLAUDE.md)'s "What this repo is" section for how the three paths
relate.

---

## Path 1: own harness (`eval/`)

```bash
python eval/run_benchmark.py --problem <op_type> --dataset <dataset> --model <model>
```

`run_benchmark.py` provisions/reuses an instance, syncs the repo, starts an
MCP session against `mcp_app/server.py` on it (`eval/mcp_client.py`), and runs
the litellm agent loop until the model stops or `--max-turns` is hit.

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

**Override ISA (e.g. Graviton3 SVE), or run the `portable` C/C++-only ablation:**
```bash
python eval/run_benchmark.py --all --dataset simd-loop --model anthropic/claude-opus-4-8 \
    --isa sve
python eval/run_benchmark.py --all --dataset simd-loop --model anthropic/claude-opus-4-8 \
    --isa portable
```

**Run simd-loop dataset:**
```bash
python eval/run_benchmark.py --problem loop_001 --dataset simd-loop --model anthropic/claude-opus-4-8
```

### All options

| Flag | Default | Description |
|------|---------|-------------|
| `--problem <name>` | — | Definition name or op_type prefix (e.g. `conv2d`) |
| `--all` | — | Run all definitions for the dataset (mutually exclusive with `--problem`) |
| `--dataset` | `ncnn` | Dataset to benchmark: `ncnn`, `simd-loop`, or `llama.cpp` |
| `--model` | (required) | LiteLLM model string, e.g. `anthropic/claude-opus-4-8` |
| `--isa` | `sve2` | ISA target: `neon`, `sve`, `sve2`, `sme2`, `portable` (plain C/C++, no SIMD intrinsics allowed) |
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
| `neon` / `portable` | `c7g.large` | Graviton3, 128-bit NEON only |
| `sve` | `c7g.large` | Graviton3, Neoverse V1, 256-bit SVE |
| `sve2` | `c8g.large` | Graviton4, Neoverse V2, 128-bit SVE2 (default) |

### Provision & teardown

```bash
python eval/provision.py --isa sve2       # Graviton4 c8g.large (SVE2=128-bit)
python eval/provision.py --isa sve        # Graviton3 c7g.large (SVE=256-bit)
python eval/provision.py --teardown
```

Or pass `--provision`/`--teardown` directly to `run_benchmark.py` to do it
automatically around a run.

---

## Path 2: MCP server for an external agent harness

Start `mcp_app/server.py` directly and connect any MCP-capable agent harness
to it — this repo doesn't drive the model in this mode.

```bash
# Local / stdio (harness and server share a host):
python3 -m mcp_app.server --dataset ncnn --author claude-code --isa sve \
    --run-dir ~/arm-bench/agent-runs-mcp/claude-code --transport stdio \
    --baseline-author baseline-ncnn-arm

# Remote / streamable-http (server on a provisioned Graviton instance,
# reached over an SSH local-port-forward — see skills/launch/ below):
python3 -m mcp_app.server --dataset ncnn --author claude-code --isa sve2 \
    --run-dir ~/arm-bench/agent-runs-mcp/claude-code --transport streamable-http
```

For the remote case, `skills/launch/launch_session.py` automates
provisioning + syncing + starting the server + printing the tunneled
endpoint in one step:

```bash
python3 skills/launch/launch_session.py launch \
    --isa sve2 --dataset ncnn
```

See [`mcp_app/README.md`](mcp_app/README.md) for the server's tool surface
and [`skills/README.md`](skills/README.md) for the full `launch`/`provision`/
`prepare-session`/`sync-results`/`teardown` command surface, plus each
harness's own skill doc (e.g.
[`skills/nanobot/nanobot-kernel-session/README.md`](skills/nanobot/nanobot-kernel-session/README.md))
for wiring the printed endpoint into that harness's MCP config.

---

## Local harness (`bench/`, no agent, no SSH)

The library every path above calls into. Useful for validating a solution
JSON you already have, on any machine:

```bash
python -m bench.cli list-definitions
python -m bench.cli bench --definition <definition> --solution <solution>
```

## Sync local codebase with remote instance

```bash
./sync_remote.sh                              # rsync the repo
./sync_remote.sh --mirror                     # force mirror (rm remote, then copy)
HOST=1.2.3.4 ./sync_remote.sh                 # different instance
```
