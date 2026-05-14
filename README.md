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