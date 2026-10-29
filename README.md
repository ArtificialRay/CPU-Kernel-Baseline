# CPU-Kernel-Baseline

A benchmark for LLM agents that write optimized AArch64 SIMD kernels (NEON, SVE,
SVE2, SME2). Each task is a kernel with a reference implementation, workloads
taken from real models, and an expert implementation to beat. The agent compiles,
tests and profiles its kernel on the target machine through MCP tools. A kernel
scores only if it is correct on every workload; its score is the speedup over the
expert implementation.

| Source | Kernels | Definitions |
|---|---|---|
| [ncnn](https://github.com/Tencent/ncnn) | Conv2D, depthwise Conv2D, pooling (FP32, INT8) | 23 |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | GEMM (FP32 to 4-bit), RMSNorm, attention, MoE | 44 |
| [Arm SIMD Loops](https://gitlab.arm.com/architecture/simd-loops) | standalone loops `loop_001` … `loop_223` | 47 |
| [KleidiAI](https://gitlab.arm.com/kleidi/kleidiai) | GEMM and convolution micro-kernels | 10 |

The tasks live in the Hugging Face dataset
[`arm-bench/arm-bench-trace`](https://huggingface.co/datasets/arm-bench/arm-bench-trace).
Its `TASKS.md` lists, for every definition, the expert baseline on each hardware
tier and whether it runs there.

## Install

Python 3.12 or later, and `clang++` (Apple clang works on macOS).

```bash
git clone https://github.com/ArtificialRay/CPU-Kernel-Baseline && cd CPU-Kernel-Baseline
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
hf download arm-bench/arm-bench-trace --repo-type dataset --local-dir bench-trace
```

Every tool reads the dataset from `bench-trace/`. Check that it loaded:

```console
$ python -m bench.cli summary
{
  "definitions": 124,
  "solutions": 417,
  "workloads": 1705,
  ...
```

## Run one kernel locally

`bench.cli` compiles a solution, checks it against the reference on every
workload and times it. No AWS account or agent is involved.

```console
$ python -m bench.cli bench --definition loop_001 --solution reference_loop_001
  6bfe094b  PASSED  min=9.08us  p5=9.15us  max_abs=0.00e+00  max_rel=0.00e+00
  06735e72  PASSED  min=43.30us  p5=43.82us  max_abs=0.00e+00  max_rel=0.00e+00
  ...
```

Kernels run only on hardware that has their instructions: scalar and NEON code on
any AArch64 machine, SVE on Graviton3, SVE2 on Graviton4, SME2 on Apple M4. An SVE
kernel run on a Mac fails with `RUNTIME_ERROR ... SIGILL`.
`python -m bench.cli list-definitions` and `list-solutions` show what there is to run.

## Run an agent

`test_scripts/bench_fleet.py` starts an EC2 instance for the ISA, runs the agent on
every definition you name, and copies the results back:

```bash
python3 test_scripts/bench_fleet.py --harness claude-code --model claude-opus-4-8 \
    --dataset simd-loop --isa sve --definitions "loop_001 loop_002"
```

| `--isa` | Instance |
|---|---|
| `neon`, `sve` | c7g.xlarge (Graviton3) |
| `sve2` | c8g.xlarge (Graviton4) |
| `sme2` | mac-m4.metal (Apple M4) |

`--dataset` is `ncnn`, `simd-loop`, `llama.cpp` or `kleidiai`. Without
`--definitions` every definition of the dataset runs. `--max-iterations` (default 40)
caps the compile, evaluate and disassemble calls per kernel. `--help` lists the rest.

| Harness | `--harness` | Needs |
|---|---|---|
| Claude Code | `claude-code` | `claude` CLI, logged in |
| Codex | `codex` | `codex` CLI, logged in |
| Cline | `cline` | `cline` CLI, `--model`, an endpoint in `skills/cline/cline-kernel-session/config.json` |
| nanobot | `nanobot` | `nanobot` CLI with a `~/.nanobot/workspace`, an API key in `skills/nanobot/nanobot-kernel-session/config.json` |
| Built-in tool-call loop | `own` | `--model` (any [litellm](https://docs.litellm.ai/) model id) |
| One completion, no tools | `single-shot` | `--model` |

### AWS setup (once)

1. An AWS account in `us-west-2` (the AMIs are pinned to that region), and an
   existing security group that allows SSH.
2. `~/.ssh/id_rsa` and `~/.ssh/id_rsa.pub`; the public key is registered with EC2.
3. `cp provisioning/workspaces.json.example provisioning/workspaces.json` and fill in
   the one entry: `account_id`, `namespace`, `aws_profile`, `aws_region`,
   `security_group_id`.
4. `terraform -chdir=terraform init`

Instances are spot by default (`--on-demand` to change). A Linux instance shuts
itself down within about 2 hours of its driver stopping (`--watchdog-minutes`);
`python3 provisioning/provision.py --teardown` removes instances right away.

An M4 needs a Dedicated Host, which AWS bills for at least 24 hours. Provisioning
never allocates one. Allocate it yourself, and release it when done:

```bash
aws ec2 allocate-hosts --instance-type mac-m4.metal --availability-zone <az> --quantity 1
aws ec2 release-hosts --host-ids <host-id>
```

### Results

Each kernel gets a directory `agent-runs-<author>/<definition>/`, where `<author>` is
`<harness>-<model>-<isa>`:

- `v1.cpp`, `v2.cpp`, …: every version the agent compiled;
- `trajectory.jsonl`: one line per tool call. An `evaluate` that passed every workload
  carries `time_speedup_geomean`, the geometric mean over workloads of
  baseline time / kernel time.

A kernel's result is the highest `time_speedup_geomean` among its passing evaluates. `--wandb`
also logs each run to Weights & Biases (fields described in
[`analysis/README.md`](analysis/README.md)).

## Other ways to use it

- **Your own agent harness:** `python3 skills/launch/launch_session.py launch --isa <isa> --dataset <dataset> --author <name>`
  starts the MCP server on an instance and prints its endpoint. Tools:
  [`mcp_app/README.md`](mcp_app/README.md); commands: [`skills/README.md`](skills/README.md).
- **Inside llama.cpp:** splice agent GEMM kernels into llama.cpp and measure tokens/s and
  perplexity: [`scripts/e2e/README.md`](scripts/e2e/README.md).
- **Add a kernel:** the `gen-definition`, `gen-workload`, `gen-candidate-solution` and
  `gen-baseline-solution` skills in [`.claude/skills/`](.claude/skills/) walk an agent
  through it.

## More documentation

- [`docs/configuration.md`](docs/configuration.md): every config file, environment
  variable and the timing protocol
- [`eval/README.md`](eval/README.md): the built-in agent loop and single-shot mode
- [`CLAUDE.md`](CLAUDE.md): how the code is organized

## License

See [`LICENSE.md`](LICENSE.md).
