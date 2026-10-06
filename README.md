# CPU-Kernel-Baseline

Evaluates LLMs on writing optimized AArch64 SIMD kernels (Neon / SVE / SVE2 /
SME2), scored against production and expert implementations from four sources:

| Source | Kernels | Precisions | Definitions |
|---|---|---|---|
| [ncnn](https://github.com/Tencent/ncnn) | Conv2D (compute bound); Conv2D Depthwise, Pooling (memory bound) | FP32, INT8 (`w8a8ch`) | 23 |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | GEMM (compute bound); RMSNorm (memory bound); MHA, GQA, MLA (prefill and decode), MoE (fused) | FP32, BF16, INT8 (`q8_0`), 4- to 6-bit (`q4_k_m`, ggml `q4_K` / `q5_K` / `q6_K`) | 44 |
| [Arm SIMD Loops](https://gitlab.arm.com/architecture/simd-loops) | Standalone loops (`loop_001` … `loop_223`) | per loop | 47 |
| [KleidiAI](https://gitlab.arm.com/kleidi/kleidiai) | GEMM, Conv2D, Conv2D Depthwise micro-kernels | FP32, F16, BF16, INT4 (`qs4c32`, `qs4cx`) | 10 |

Counts are for dataset revision `def2c3e` (2026-09-26);
`python -m bench.cli list-definitions` prints what your copy contains. 18 of
the llama.cpp GEMMs (`gemm_ggml_*`) are the quantized matmul shapes of
Qwen3.5-4B and Llama-3.1-8B, used for end-to-end runs.

ncnn and llama.cpp kernel shapes and workloads are extracted from real models:
resnet50, mobilenetv3-large, alexnet, googlenet, squeezenet1_1, vgg16 (ncnn);
qwen1.5-moe-a2.7b, olmoe-1b-7b, deepseek-v3, llama-3.1-8b, mistral-7b-v0.1,
qwen3.5-4b (llama.cpp).

kernel dataset (bench-trace):
https://huggingface.co/datasets/arm-bench/arm-bench-trace

---

## Prerequisites

Install the dependencies, then download the kernel dataset into `bench-trace/`
at the root of your local repository (every tool here reads it from there):

```bash
pip install -r requirements.txt
huggingface-cli download arm-bench/arm-bench-trace --repo-type dataset --local-dir bench-trace
```

Provisioning and remote runs need an AWS account, Terraform, an SSH key, and
the config files described under [Configuration](#configuration) below
(start with `provisioning/workspaces.json`).

## Configuration

There are a lot of config files, but they fall into three groups and only the
first one is something you have to set up yourself.

### 1. Set up once (git-ignored — copy each from the `.example` next to it)

| File | What goes in it | Needed for |
|---|---|---|
| `provisioning/workspaces.json` | **Exactly one** entry: the Terraform workspace this checkout uses. The key is the workspace name (selected automatically — do **not** set `TF_WORKSPACE`). Fields: `account_id` (AWS account guard), `namespace` (suffix for AWS resource names), `aws_profile`, `aws_region`, `security_group_id` (an *existing* security group that allows SSH — Terraform does not create it). This is the single source for AWS account/region/profile: don't repeat any of it in `.env`. | any provisioning / remote run |
| `eval/llm_providers.json` | Per-provider `api_key` / `api_base` for the litellm loop; anything left out falls back to the provider's usual environment variable. | `--harness own` |
| `skills/nanobot/nanobot-kernel-session/config.json` | nanobot's model, provider + API key, and MCP server wiring. The one exception here: it is checked in as the default, with every key empty. Fill in your provider's `apiKey`, or keep the key out of git by pointing `NANOBOT_CONFIG_BASE` at a copy. | `--harness nanobot` |
| `skills/codex/codex-kernel-session/config.json` | Optional custom endpoint (`base_url`, `api_key`); without it codex uses the account `codex login` set up. | `--harness codex` |
| `skills/cline/cline-kernel-session/config.json` | Custom endpoint (`base_url`, `api_key`) — required, cline has no login to fall back to. | `--harness cline` |

`--harness claude-code` uses the account `claude login` set up.
It runs every job in its own empty directory outside this repo, with
`--permission-mode dontAsk`: the agent gets this session's MCP tools plus file
tools confined to that directory, and auto-memory is off. It cannot read
`bench-trace/` (hidden workloads, expert solutions), and nothing carries over
from one job to the next.

`.env` (from `.env.example`) is **optional** and only for one-off overrides of
the above, e.g. `NANOBOT_CONFIG_BASE` to try another nanobot config. A value
there wins over the file it overrides.

### 2. Checked-in defaults (edit only to change how kernels are evaluated or built)

| File | Purpose |
|---|---|
| `config/kernel_contracts.yaml` | Kernel evaluation parameters: op-type correctness/timing overrides, disallowed source patterns, ISA→march mapping, baseline authors |
| `config/dataset_builds.json` | Step-by-step clone/build of each dataset's native lib (ncnn, ggml) on a remote instance |
| `config/rsync_allowlist.json` | Repo paths synced to instances before a session (override once with `RSYNC_ALLOWLIST` in `.env`) |

### 3. Written by the tools (do not edit by hand)

| File | Purpose |
|---|---|
| `provisioning/eval_config.json` | The instances currently up (host, user, key). Created and updated by `provision.py`; `eval_config.json.example` only shows the format. |
| `terraform/terraform.tfstate.d/` | Local Terraform state, one directory per workspace |
| `agent-runs-<author>/`, `harness_trajs/` | Per-kernel trajectories and logs synced back from the instances |

### Notes on remote instances

- **Python environment:** every instance (Graviton and Mac alike) runs `bench/`
  and `mcp_app/` from one uv-managed venv at `~/venv`, built from
  `requirements.txt` by `provision.py`. Nothing to set up by hand, and a reused
  instance is checked (and repaired if a package is missing) before a run.
- **Mac (`mac-m4.metal`) needs a Dedicated Host.** Provisioning picks an
  existing `available` host with no instance on it and never allocates one
  itself (AWS bills a 24-hour minimum). Allocate one first:
  `aws ec2 allocate-hosts --instance-type mac-m4.metal --availability-zone <az> --quantity 1`.
  A host that just lost its instance stays `pending` for a while while AWS
  scrubs it.

### Optional environment variables

Only needed in specific cases, set in `.env` or the shell.

| Variable | Effect |
|---|---|
| `NCNN_ROOT`, `LLAMA_CPP_ROOT` | Point the local `bench/` harness at an existing ncnn / llama.cpp checkout |
| `ARMBENCH_NOTEPAD=1` | Give the `own` harness's agent a persistent scratchpad tool |
| `ARMBENCH_CC_JOB_ROOT` | Parent of the per-job working directories `--harness claude-code` creates (default: the system temp dir); must be outside this repo |
| `OPENROUTER_API_KEY` | Used by `scripts/bench_loop_agent.py` |
| `WANDB_INSTANCE_TYPE` | Label recorded on runs logged with `--wandb` |

### Timing protocol

`eval_defaults` in `config/kernel_contracts.yaml` applies to every target:

```yaml
eval_defaults:
  warmup: 10
  repeat: 50
  inner_iters: auto
  target_sample_ns: 10000000
```

With `inner_iters: auto`, each of the `repeat` timed samples calls the kernel
back to back and reports the time per call; the fastest sample is the result.
A baseline's count is probed so that one sample lasts about
`target_sample_ns` (10 ms), and a candidate is timed with its baseline's
count. Timing a single call per sample instead is noisy for calls shorter
than about 1 ms, most of all on Apple Silicon.

Every trace records its `inner_iters` and a `timing_protocol` string built
from the four values above. A baseline counts for a candidate only when both
carry the same `timing_protocol`, so measurements taken another way (for
example with the earlier defaults, `warmup: 5` and `inner_iters: 1`) are
never mixed in. A session re-collects such a baseline before it scores
anything against it, which means that changing any of the four values
re-collects the baselines on every instance. With `bench.cli`, run
`collect-baselines` again; until then `bench` reports no speedup rather than
one against the old baseline.

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

Our project use terraform to initialize and provision AWS instances, we support two type instances: `graviton` instances and `macm4.metal`.

At the first time to use our project tools, please init your terraform with:

```bash
cd terraform/
terraform init
```

One parametrized entry point for driving a batch kernel-optimization run
against any of this repo's harnesses — provisions/reuses an instance, starts
an `mcp_app` session, runs every matching definition through the chosen
harness with per-job retry/logging, syncs results back, then closes the
session once every job's local trajectory is confirmed complete.

```bash
python3 test_scripts/bench_fleet.py --harness claude-code \
    --dataset ncnn --isa sve2 --model anthropic/claude-opus-4-8
```

Use `--definitions` to control which kernels the agent optimizes: one name, a
space-separated list, or a JSON array (quote it, so the shell passes it through
unchanged). Without `--definitions`, the entrypoint runs every definition in
that dataset except the end-to-end ones (tagged `e2e:<model>`), which run only
when named.

```bash
python3 test_scripts/bench_fleet.py --harness nanobot \
    --dataset ncnn --isa sve --definitions "conv2d_fp32_kh3_kw3_sh1_sw1_dh1_dw1_p1"
python3 test_scripts/bench_fleet.py --harness own --model anthropic/claude-opus-4-8 \
    --dataset llama.cpp --isa sve2 \
    --definitions '["gemm_q4_k_m_n2048_k1536","gemm_q4_k_m_n2048_k2048","gemm_q8_0_n1024_k2048","gemm_q8_0_n1408_k2048","gemm_q8_0_n2048_k1024"]'
```

Each harness's own `HarnessAdapter` lives in its own module under `test_scripts/harness_adapters/`. Run
`python3 test_scripts/bench_fleet.py --help` for the full flag reference
(`--definitions`, `--min-iterations`/`--max-iterations`, `--retries`,
`--sync-solutions`, `--on-demand`, `--until-complete`, `--wandb`, ...).
`--dataset` is one of `ncnn`, `simd-loop`, `llama.cpp`, `kleidiai`; `--isa` is
one of `neon`, `sve`, `sve2`, `sme2`, `portable` (plain C/C++, no SIMD
intrinsics).

`test_scripts/run_driver_smoke.sh` is a separate, narrower smoke-test:
compile/evaluate/disassemble/submit against a couple of reference-scalar
kernels per dataset, no LLM involved.

---
## Run the benchmark with supported harness

| Harness | `--harness` value | Requires |
|---|---|---|
| Claude Code | `claude-code` | `claude` CLI on PATH |
| Codex | `codex` | `codex` CLI on PATH |
| Cline | `cline` | `cline` CLI on PATH, `--model`, and an endpoint in `skills/cline/cline-kernel-session/config.json` |
| nanobot | `nanobot` | `nanobot` CLI on PATH + a bootstrapped `~/.nanobot/workspace` |
| This repo's own loop | `own` | `--model` (no external CLI) |
| Single shot, no tools | `single-shot` | `--model` (no external CLI) |

`--harness nanobot` reads its base config from
`skills/nanobot/nanobot-kernel-session/config.json` by default. Passing
`--model` alone only overrides `agents.defaults.model` — the provider (and
its API key) still comes from that checked-in config, so switching to a
model from a different provider needs its own base config. Set
`NANOBOT_CONFIG_BASE` in `.env` (see `.env.example`) to point at one instead.

### Supported harness (claude-code / codex / cline / nanobot / own / single-shot)

If your harness already has a `HarnessAdapter`
(`test_scripts/harness_adapters/`), you can use `test_scripts/bench_fleet.py` (see
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

**--model** is a required argument for own harness as there are no default model provided for own harness

`--harness single-shot` uses the same in-process path
(`eval/single_shot.py::run_single_shot`) but gives the model one completion
with no tools, `--samples` times per definition, and measures each result
through the same MCP `compile`/`evaluate`.

See [`eval/README.md`](eval/README.md) for the agent-loop and single-shot
details, `provisioning/provision.py`'s standalone provisioning commands, and where
results/traces end up.

---

### Custom MCP server session

For a harness that isn't one of the supported adapters (or for
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

Always remember to teardown MCP server after your session:
```bash
python3 skills/launch/launch_session.py teardown
```
use `--label` to specify the exact instance you want to terminate at provisioning/eval_config.json, otherwise, it will terminate all living instances


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
