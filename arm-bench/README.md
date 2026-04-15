# armbenchmark

A KernelBench-style LLM benchmark for Arm SIMD kernel generation. Contains 75 real-world loop kernels (NEON / SVE / SVE2 / SME2) packaged as problems for evaluating LLMs on their ability to write optimized AArch64 code.

Built on top of [SIMD Loops](https://gitlab.arm.com/architecture/simd-loops) (BSD-3-Clause).

## How it works

Each problem gives the LLM a scalar C implementation (and a NEON reference when available) and asks it to produce a faster SIMD version. Evaluation runs on an AWS Graviton instance over SSH — the LLM can call tools iteratively before submitting a final solution.

```
LLM gets:  scalar C function + NEON reference (when available) + ISA target (SVE / SVE2)

LLM tools:
  compile(code)       → build errors / warnings
  run(n)              → checksum correct? + timing
  perf(n)             → cycles, IPC, L1D miss rate
  disassemble(fn)     → generated AArch64 assembly
  submit(code)        → final score

Score:  Level 0  compile fail / wrong checksum
        Level 1  correct
        Level 2  correct + faster than scalar
        Level 3  correct + faster than autovec  ← "fast_p"
        Level 4  correct + faster than hand-written SVE/SVE2 reference
```

**55 SVE problems** (Graviton3, `c7g.large`) · **20 SME2 problems** (future — see below)

## Quickstart

### 1. Install dependencies

Python:
```bash
pip install litellm
```

System packages (required on any host that actually runs `SIMDTools.compile()` — i.e. locally when `handle=None`, or on the remote when using `--provision`):

```bash
wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | sudo tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc
echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-18 main" | sudo tee /etc/apt/sources.list.d/llvm-18.list
sudo apt-get update -y
sudo apt-get install -y cmake clang-18 llvm-18 libomp-18-dev g++
sudo ln -sf /usr/bin/clang++-18 /usr/local/bin/clang++
sudo ln -sf /usr/bin/llvm-objdump-18 /usr/local/bin/llvm-objdump-18
```

### 2. Provision an instance

```bash
python eval/provision.py --isa sve2
# Runs terraform apply, waits for cloud-init, rsyncs source.
# Writes connection info to eval/eval_config.json automatically.
```

Requires: AWS credentials in environment, Terraform installed, `~/.ssh/id_rsa` key pair.

### 3. Verify the pipeline (optional)

```bash
python -m eval.test_workflow --isa sve2
```

Injects a known-good scalar candidate for `loop_001`, then exercises compile → run → perf → disassemble end-to-end. Useful after first provisioning to confirm SSH, build, and PMU access all work.

### 4. Collect baselines (run once)

```bash
python scripts/collect_baselines.py --isa sve    # c7g (Graviton3, SVE)
python scripts/collect_baselines.py --isa sve2   # c8g (Graviton4, SVE2)
```

Builds scalar, autovec, and hand-written ISA reference targets; records timings to `baselines/{tier}.json` (scalar_ms, autovec_ms, ref_ms).

### 5. Run the benchmark

**Agentic mode** (LLM uses tools iteratively):
```bash
python eval/run_benchmark.py --problem loop_001 --isa sve --model anthropic/claude-opus-4-6
python eval/run_benchmark.py --all --isa sve --model anthropic/claude-opus-4-6 --teardown
```

**Single-shot mode** (KernelBench-compatible, generate then evaluate separately):
```bash
python eval/generate_samples.py --all --isa sve2 --model openai/gpt-4o
python eval/eval_from_generations.py --all --isa sve2
```

### 6. Teardown

```bash
python eval/provision.py --teardown
```

## Repository layout

```
dataset/problems/           75 problem.py files (scalar code + metadata)
dataset/problems.json       Problem index
eval/                       Benchmark harness (provision, tools, evaluator, CLIs)
  eval/provision.py         Terraform wrapper — spin up / tear down Graviton instances
  eval/tools.py             SSH-backed tool calls (compile, run, perf, disassemble, submit)
  eval/evaluator.py         Agentic eval loop (builds prompt, drives LLM tool calls)
  eval/config.py            Problem loader (extracts scalar + NEON reference code)
  eval/test_workflow.py     End-to-end smoke test — no LLM, validates the full pipeline
  eval/eval_config.json.example   SSH config template — copy and fill in
loops/                      Kernel source (C + inline asm, all ISA variants)
scripts/                    Dataset extraction + baseline collection
  scripts/fix_candidate_guards.py   Wraps bare inner_loop_NNN definitions in
                                    #if !defined(HAVE_CANDIDATE) guards (maintenance)
terraform/                  EC2 provisioning (Graviton3/4 spot instances)
baselines/                  Timing baselines written by collect_baselines.py
generations/                LLM outputs written by generate_samples.py
results/                    Scored results written by eval scripts
```

## Build targets (for running kernels directly)

Requires clang-18 on AArch64 Linux. On AWS: `terraform/setup.sh` installs it.

| Target | ISA | Instance |
|--------|-----|----------|
| `scalar` / `c-scalar` | scalar C | any |
| `neon` | NEON 128-bit | any AArch64 |
| `sve` | SVE 256-bit | c7g (Graviton3) |
| `sve2` | SVE2 128-bit | c8g (Graviton4) |
| `sme2` | SME2 | — (no AWS support yet) |
| `autovec-sve` | compiler autovec (SVE) | c7g |
| `autovec-sve2` | compiler autovec (SVE2) | c8g |

```bash
make sve2
./build/sve2/bin/simd_loops -k 1 -n 100
```

## SME2 — future target

The dataset includes 20 SME2 kernel problems sourced from the upstream SIMD Loops project. However, **no AWS EC2 instance currently supports SME2** — Graviton4 (Neoverse V2, c8g) implements SVE2 but not the Scalable Matrix Extension. The first publicly available SME2 hardware is Apple M4.

The SME2 problems are kept in the dataset for future use. Attempting to run `--isa sme2` in the benchmark or collect_baselines scripts will raise an error until hardware support is available.

## License

BSD 3-Clause — see [LICENSE.md](LICENSE.md).
