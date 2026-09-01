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

- **Own harness** — this repo's own litellm agent loop (`eval/evaluator.py`),
  driven via `test_scripts/bench_fleet.py --harness own`: provisions a
  Graviton instance, starts `mcp_app/server.py` on it, and runs a
  self-contained tool-call loop against it. No external agent harness needed.
- **MCP server for an external harness** — start `mcp_app/server.py` directly
  (or via `skills/launch/`) and point an external agent harness (nanobot,
  Claude Code, ...) at it. This repo never drives the model in this mode; the
  external harness does.

Both modes share the same `compile`/`evaluate`/`disassemble`/`submit` tool
surface and the same local `bench/` library underneath — see
[CLAUDE.md](CLAUDE.md)'s "What this repo is" section for how the three paths
relate.

---

## Batch driver (`test_scripts/bench_fleet.py`)

One parametrized entry point for driving a batch kernel-optimization run
against any of this repo's harnesses — provisions/reuses an instance, starts
an `mcp_app` session, runs every matching definition through the chosen
harness with per-job retry/logging, syncs results back, then closes the
session once every job's local trajectory is confirmed complete.

```bash
python3 test_scripts/bench_fleet.py --harness claude-code \
    --dataset ncnn --isa sve2 --model anthropic/claude-opus-4-8
python3 test_scripts/bench_fleet.py --harness nanobot \
    --dataset ncnn --isa sve --definitions "conv2d_fp32_kh3_kw3_sh1_sw1_dh1_dw1_p1"
python3 test_scripts/bench_fleet.py --harness own \
    --dataset ncnn --isa sve2 --model anthropic/claude-opus-4-8
```

`--harness` selects `claude-code` / `nanobot` / `own` (Path 1 — this repo's
own litellm loop, in-process, no external CLI); each harness's own
`HarnessAdapter` lives in `test_scripts/harness_adapters.py`. Run
`python3 test_scripts/bench_fleet.py --help` for the full flag reference
(`--definitions`, `--min-iterations`/`--max-iterations`, `--retries`,
`--sync-solutions`, `--on-demand`, ...).

`test_scripts/run_driver_smoke.sh` is a separate, narrower smoke-test:
compile/evaluate/disassemble/submit against a couple of reference-scalar
kernels per dataset, no LLM involved.

---

## Path 1: own harness (`eval/`)

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
