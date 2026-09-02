# CPU-Kernel-Baseline

Evaluates LLMs on writing optimized AArch64 SIMD kernels for ncnn / llama.cpp /
synthetic simd-loop benchmarks. 

FP32 Kernels available:
| Kernel Name | Type | Source |
|---|---|---|
| RMSNorm | Memory Bound | llama.cpp |
| Conv2D Depthwise | Memory Bound | ncnn |
| Pooling (Reduction) | Memory Bound | ncnn |
| Conv2D | Compute Bound | ncnn |
| GEMM | Compute Bound | ncnn |

BF16 Kernels available:
| Kernel Name | Type | Source |
|---|---|---|
| GEMM | Compute Bound | llama.cpp |
| MHA | Fused | llama.cpp |
| GQA | Fused | llama.cpp |
| MLA (both prefill and decode) | Fused | llama.cpp |
| MoE | Fused | llama.cpp |

INT8 Kernels available:
| Kernel Name | Type | Source |
|---|---|---|
| Conv2D Depthwise | Memory Bound | ncnn (w8a8ch) |
| Conv2D | Compute Bound | ncnn (w8a8ch) |
| GEMM | Memory Bound/Compute Bound | ncnn (w8a8ch), llama.cpp (q8_0) |
| MoE | Fused | llama.cpp (q8_0) |

INT4 Kernels available:
| Kernel Name | Type | Source |
|---|---|---|
| GEMM | Compute Bound | llama.cpp (q4_k_m) |
| MoE | Fused | llama.cpp (q4_k_m) |


Kernel definitions are extracted from real model architectures: qwen1.5-moe-a2.7b, olmoe-1b-7b, deepseek-v3, llama-3.1-8b, mistral-7b-v0.1, resnet50, mobilenetv3-large, alexnet, googlenet, squeezenet1_1, vgg16, deepspeech2

kernel dataset (bench-trace):
https://huggingface.co/datasets/arm-bench/arm-bench-trace

---

## Prerequisites

Download kernel dataset to the main directory of your local repository, then run dependency installation:

```bash
pip install -r requirements.txt
```

Provisioning and remote runs need an AWS account with Terraform configured
(`terraform/`) and an SSH key. See `eval/eval_config.json.example`.

## Two ways to run an agent against this benchmark

- **MCP server for an external harness** — start `mcp_app/server.py` directly
  (or via `skills/launch/`) and point an external agent harness (nanobot,
  Claude Code, ...) at it. This repo never drives the model in this mode; the
  external harness does.
- **Own harness** — this repo's own litellm agent loop (`eval/evaluator.py`),
  driven via `test_scripts/bench_fleet.py --harness own`: provisions a
  Graviton instance, starts `mcp_app/server.py` on it, and runs a
  self-contained tool-call loop against it. No external agent harness needed.

Both modes share the same `compile`/`evaluate`/`disassemble`/`submit` tool
surface and the same local `bench/` library underneath — see
[CLAUDE.md](CLAUDE.md)'s "What this repo is" section for how the three paths
relate.

---

## Benchmarking Entrypoint(`test_scripts/bench_fleet.py`)

One parametrized entry point for driving a batch kernel-optimization run
against any of this repo's harnesses — provisions/reuses an instance, starts
an `mcp_app` session, runs every matching definition through the chosen
harness with per-job retry/logging, syncs results back, then closes the
session once every job's local trajectory is confirmed complete.

```bash
python3 test_scripts/bench_fleet.py --harness claude-code \
    --dataset ncnn --isa sve2 --model anthropic/claude-opus-4-8
```

Use `--definitions` to control what kernel you'd like agent to optimize, you can add one or multiple kernels if you wish. If `--definitions` is not parsed, the entrypoint will run all definitions in that dataset

```bash
python3 test_scripts/bench_fleet.py --harness nanobot \
    --dataset ncnn --isa sve --definitions "conv2d_fp32_kh3_kw3_sh1_sw1_dh1_dw1_p1"
python3 test_scripts/bench_fleet.py --harness own \
    --dataset llama.cpp --isa sve2 --definitions ["gemm_q4_k_m_n2048_k1408","gemm_q4_k_m_n2048_k2048","gemm_q8_0_n1024_k2048","gemm_q8_0_n1408_k2048","gemm_q8_0_n2048_k1024"]
```

Each harness's own `HarnessAdapter` lives in `test_scripts/harness_adapters.py`. Run
`python3 test_scripts/bench_fleet.py --help` for the full flag reference
(`--definitions`, `--min-iterations`/`--max-iterations`, `--retries`,
`--sync-solutions`, `--on-demand`, ...).

`test_scripts/run_driver_smoke.sh` is a separate, narrower smoke-test:
compile/evaluate/disassemble/submit against a couple of reference-scalar
kernels per dataset, no LLM involved.

---
## Run the benchmark with supported harness

| Harness | `--harness` value | Requires |
|---|---|---|---|---|
| Claude Code | `claude-code` | `claude` CLI on PATH |
| nanobot | `nanobot` | `nanobot` CLI on PATH + a bootstrapped `~/.nanobot/workspace` |
| This repo's own loop | `own` | none (no external CLI) |

### Supported harness (claude-code / nanobot / own)

If your harness already has a `HarnessAdapter`
(`test_scripts/harness_adapters.py`), you can use `test_scripts/bench_fleet.py` (see
"Benchmarking Entrypoint" above) directly, it provisions the instance, starts the MCP
session, and drives the harness end to end in one command:

```bash
python3 test_scripts/bench_fleet.py --harness claude-code \
    --dataset ncnn --isa sve2 --model anthropic/claude-opus-4-8
```

### own harness (`eval/`)

```bash
python3 test_scripts/bench_fleet.py --harness own \
    --dataset <dataset> --isa <isa> --model <model>
```

`bench_fleet.py --harness own` provisions/reuses an instance, syncs the
repo, starts an MCP session against `mcp_app/server.py` on it
(`eval/mcp_client.py::attach()`), and runs the litellm agent loop
(`eval/evaluator.py::run_agentic_eval`) in-process for every definition
matching `--dataset` (narrow with `--definitions`) until the model stops or
`--max-iterations` is hit.

See [`eval/README.md`](eval/README.md) for `eval/evaluator.py`'s agent-loop
details, `eval/provision.py`'s standalone provisioning commands, and where
results/traces end up.

---

### Custom MCP server session

For a harness that isn't one of the three supported adapters (or for
debugging the MCP surface directly), specify your own `--dataset`,
`--author`, and `--isa` and start the session with
`skills/launch/launch_session.py`:

```bash
python3 skills/launch/launch_session.py launch \
    --isa sve2 --dataset ncnn --author <you-specified-author>
```

See [`mcp_app/README.md`](mcp_app/README.md) for the server's tool surface
and [`skills/README.md`](skills/README.md) for the full `launch`/`provision`/
`prepare-session`/`sync-results`/`teardown` command surface.

To enable your agent know how to use MCP tools, please refer to harness's own skill doc (e.g. [`skills/nanobot/nanobot-kernel-session/SKILL.md`](skills/nanobot/nanobot-kernel-session/SKILL.md)) and the harness's MCP config wiring guideline (e.g. [`skills/nanobot/nanobot-kernel-session/README.md`](skills/nanobot/nanobot-kernel-session/README.md))
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
