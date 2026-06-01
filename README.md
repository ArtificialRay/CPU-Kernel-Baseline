# CPU-Kernel-Baseline
CPU Kernels/Operators for vision model / LLM extracted from 5 codebases, serving as baseline for the project

sync local codebase with remote instance
**add new files:**
```bash
./arm-bench/sync_remote.sh
```

**force mirror:**
```bash
./arm-bench/sync_remote.sh --mirror
```

**change another instance via ip address**
```bash
HOST=1.2.3.4 ./arm-bench/sync_remote.sh
```

## How to bench ncnn kernel with arm-bench
arm-bench is an agentic kernel optimization pipeline for arm SIMD loops. Now arm-bench supports iteratively optimize and evaluate NN operator/kernel. 

### 1. Install dependencies

```bash
pip install litellm
```
then cd to arm-bench
```bash
cd arm-bench
```

### 2. Provision an instance

For the first time to provision an instance, run
```bash
cd your-repo-dir/terraform
```
then run 
```bash
terraform init
```

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
python scripts/collect_baselines_ncnn.py --isa sve    # c7g (Graviton3, SVE)
python scripts/collect_baselines_ncnn.py --isa sve2   # c8g (Graviton4, SVE2)
```

Builds candidate and baseline targets; records timings to `baselines/{tier}.json` (candidate_ms, baseline_ms).

### 5. Run the benchmark
set `--mode ncnn` to bench ncnn kernels
**Agentic mode** (LLM uses tools iteratively):
```bash
python -m eval.run_benchmark --all --mode ncnn --isa sve --model anthropic/claude-opus-4.6
python -m eval.run_benchmark --problem conv --mode ncnn --isa sve --model anthropic/claude-sonnet-4.6
```

### 6. Teardown

```bash
python eval/provision.py --teardown
```

---

## arm-bench/bench: Walkthroughs & Collect Baseline Kernel Performance(only convolution kernel is supported)

The `bench/` package replaces the per-op `starter/` + `tests/` + `perf/` +
`candidate_src/` trio with the following schema:

| | |
|---|---|
| `definitions/<op>/<name>.json` | framework-agnostic op spec (axes, inputs, outputs, PyTorch reference) |
| `solutions/<dataset>/<author>/<op>/<name>.json` | concrete C++ kernel + compile/link spec |
| `solutions/<dataset>/_harness/<op>.{cpp,h}` | shared dataset harness (data, not bench code) |
| `workloads/<op>/<name>.jsonl` | per-Definition concrete sizes + scalar inputs |
| `traces/<op>/<name>.jsonl` | one record per benchmarked (Definition, Solution, Workload) |

The workflow to collect runtime performance of convolution kernel

### 1. Install Python deps (once per instance)

Remote (Ubuntu ARM, after `python eval/provision.py --instance c7g.large`):
```bash
ssh -i ~/.ssh/id_rsa ubuntu@<host> '
  sudo apt-get update -qq && sudo apt-get install -y -qq python3-pip clang cmake libomp-dev
  pip3 install --user --break-system-packages -r ~/arm-bench/requirements.txt
'
```

### 2. Build the full ncnn static lib on the instance (one-time per ISA)

Every baseline Solution links against a full `libncnn.a` built from the ncnn
checkout (`~/ncnn`, a sibling of `~/arm-bench`). It is required for two reasons:
it resolves Convolution_arm's whole per-ISA helper tree (`convolution_arm_i8mm.cpp`,
`..._asimddp.cpp`, `..._sve.cpp`, …, each compiled with its own `-march` flags),
and its CMake configure step generates the `platform.h` the per-solution build
includes. A hand-picked subset would leave undefined symbols at dlopen, so build
the whole thing once.

**a. Configure** (fast; also generates `build/src/platform.h`):

```bash
ssh -i ~/.ssh/id_rsa ubuntu@<host> 'cd ~/ncnn && CC=clang CXX=clang++ cmake -B build \
    -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_TESTS=OFF -DNCNN_BUILD_EXAMPLES=OFF \
    -DNCNN_BUILD_BENCHMARK=OFF -DNCNN_VULKAN=OFF -DNCNN_SHARED_LIB=OFF'
```

**b. Build the `ncnn` target only** (compiles the full library — a few minutes
on a 2-core c7g.large; scale `-j` to the instance):

```bash
ssh -i ~/.ssh/id_rsa ubuntu@<host> 'cd ~/ncnn && cmake --build build -j"$(nproc)" --target ncnn'
```

**c. Verify** the archive and generated header exist:

```bash
ssh -i ~/.ssh/id_rsa ubuntu@<host> 'ls -la ~/ncnn/build/src/libncnn.a ~/ncnn/build/src/platform.h'
# output: ~/ncnn/build/src/libncnn.a  (the lib the NcnnBuilder links against)
```

The builder auto-discovers `~/ncnn/build/src/libncnn.a` (override the checkout
location with `ARMBENCH_BASE_ROOT=<path>` if it differs). This is a one-time
cost per instance/ISA: the lib is cached and reused across every baseline build.

### 3. Collect baselines

Runs every baseline-author Solution against its Definition's workloads, caching
the timings so candidate `armbench bench` runs can derive `reference_min_ns`
and `speedup`:

```bash
ssh -i ~/.ssh/id_rsa ubuntu@<host> 'cd arm-bench && python3 -m bench.cli collect-baselines'
# → appends a PASSED trace per (Definition, Workload) to traces/conv2d/<def>.jsonl
```

Pull traces back locally for inspection / source-of-truth versioning:
```bash
rsync -avz -e "ssh -i ~/.ssh/id_rsa" ubuntu@<host>:arm-bench/bench-trace/traces/ bench-trace/traces/
```

### 4. Inspect / run a candidate Solution

Read traces at local or remote instance
```bash
cd arm-bench
python -m bench.cli summary                    # JSON: counts + pass/fail
python -m bench.cli list-definitions           # def → #solutions / #workloads
python -m bench.cli list-solutions             # all authors + datasets
```

Run bench at remote instance
```bash
ssh -i ~/.ssh/id_rsa ubuntu@<host> 'cd arm-bench && python -m bench.cli bench \
  --definition conv2d_kh3_kw3_sh1_sw1_dh1_dw1_c64_c128 \
  --solution   my-author_my-kernel '
```
my-author: reference-scalar / baseline-ncnn-arm
my-kernel: all solutions under `arm-bench/bench/reference-scalar/conv2d/` or under `arm-bench/bench/baseline-ncnn-arm/conv2d/`