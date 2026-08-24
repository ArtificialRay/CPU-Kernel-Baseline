# eval/ — own harness (in-repo litellm agent loop)

The repo provide a in-repo litellm agent loop to use mcp tools for test and fun

## Prerequisites

```bash
pip install -r requirements.txt   # from repo root
```

- A `.env` file at the repo root with the API key for whichever provider
  `--model` names (e.g. `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`) — loaded
  via `python-dotenv`.
- An AWS account with Terraform configured (`terraform/`) and an SSH key, if
  you'll be provisioning instances (not needed if you're only pointing at an
  already-running instance recorded in `eval_config.json`).
- `eval/eval_config.json` — copy from `eval/eval_config.json.example`.
  Records `host`/`user`/`key_file`/`instance_type` per label (default label
  is `f"{dataset}-{isa}"`). Shared with `skills/launch/`'s own provisioning,
  so an instance either side brought up is visible to the other.

## Quickstart

```bash
python eval/run_benchmark.py --problem <op_type> --dataset <dataset> --model <model>
```

What this does, end to end:
1. Provisions a fresh instance or reuses one already recorded in
   `eval_config.json` for this `{dataset}-{isa}` label (`eval/provision.py`).
2. Starts `mcp_app.server` on it and opens an MCP client session
   (`eval/mcp_client.py`), reused across every definition in this run.
3. Runs the litellm agent loop (`eval/evaluator.py::run_agentic_eval`) for
   each definition: the model calls `compile`/`evaluate`/`disassemble`
   against the MCP session until it stops or `--max-turns` is hit.
4. Saves the run's result to `results/` (unless `--no-save`); the best
   kernel itself is already persisted to `bench-trace/` by the MCP server on
   every new best, independent of this.
5. Tears the instance down if `--teardown` was passed.

## Usage examples

**Single definition (ncnn dataset, SVE2 / Graviton4 by default):**
```bash
python eval/run_benchmark.py --problem conv2d --dataset ncnn --model anthropic/claude-opus-4-8
```

**All definitions for a dataset:**
```bash
python eval/run_benchmark.py --all --dataset ncnn --model anthropic/claude-opus-4-8
```

**Provision a fresh instance, run, then tear it down automatically:**
```bash
python eval/run_benchmark.py --all --dataset ncnn --model anthropic/claude-opus-4-8 \
    --provision --teardown
```

**Reuse an instance already recorded in `eval_config.json`** (default —
just omit `--provision`):
```bash
python eval/run_benchmark.py --problem conv2d --dataset ncnn --model anthropic/claude-opus-4-8
```

**Override ISA (e.g. Graviton3 SVE), or run the `portable` C/C++-only ablation**
(agent-submitted code may not use NEON/SVE intrinsics; compares agent-optimized
plain C++ against hand-written SIMD):
```bash
python eval/run_benchmark.py --all --dataset simd-loop --model anthropic/claude-opus-4-8 \
    --isa sve
python eval/run_benchmark.py --all --dataset simd-loop --model anthropic/claude-opus-4-8 \
    --isa portable
```

**simd-loop / llama.cpp datasets:**
```bash
python eval/run_benchmark.py --problem loop_001 --dataset simd-loop --model anthropic/claude-opus-4-8
python eval/run_benchmark.py --all --dataset llama.cpp --model anthropic/claude-opus-4-8
```

**Quiet batch run, keep full trajectories, skip already-collected baselines:**
```bash
python eval/run_benchmark.py --all --dataset ncnn --model anthropic/claude-opus-4-8 \
    --quiet --save-trace --skip-baselines
```

## `run_benchmark.py` options

| Flag | Default | Description |
|------|---------|-------------|
| `--problem <name>` | — | Definition name or op_type prefix (e.g. `conv2d`) |
| `--all` | — | Run all definitions for the dataset (mutually exclusive with `--problem`) |
| `--dataset` | `ncnn` | Dataset to benchmark: `ncnn`, `simd-loop`, or `llama.cpp` |
| `--model` | (required) | LiteLLM model string, e.g. `anthropic/claude-opus-4-8` |
| `--isa` | `sve2` | ISA target: `neon`, `sve`, `sve2`, `sme2`, `portable` (plain C/C++, no SIMD intrinsics allowed) |
| `--provision` | off | Provision a new instance even if one is already configured for this label |
| `--teardown` | off | Destroy the instance after evaluation |
| `--max-turns` | `20` | Max agent turns per definition |
| `--quiet` | off | Suppress per-turn output |
| `--no-save` | off | Don't write this run's result to `results/` |
| `--save-trace` | off | Also save the full `version_history` to `traces/` |
| `--skip-baselines` | off | Skip lazy baseline collection (use if baselines are already present) |

### Instance types

| ISA | Instance | Notes |
|-----|----------|-------|
| `neon` / `portable` | `c7g.large` | Graviton3, 128-bit NEON only |
| `sve` | `c7g.large` | Graviton3, Neoverse V1, 256-bit SVE |
| `sve2` | `c8g.large` | Graviton4, Neoverse V2, 128-bit SVE2 (default) |

## Provisioning (`eval/provision.py`)

Standalone script — `run_benchmark.py --provision`/`--teardown` just call
into it. Useful directly when you want an instance to persist across several
`run_benchmark.py` invocations, or to check/tear down what's currently up.

```bash
python eval/provision.py --isa sve2                    # provision (label defaults to isa)
python eval/provision.py --isa sve2 --dataset ncnn      # + build ncnn's native lib right after
python eval/provision.py --status                        # show what's currently up
python eval/provision.py --teardown                       # destroy every recorded instance
python eval/provision.py --teardown --label ncnn-sve2     # destroy just one label
python eval/provision.py --isa sve2 --on-demand            # on-demand, not spot — for long unattended runs
```

| Flag | Default | Description |
|------|---------|-------------|
| `--isa` | — | ISA target: `neon`, `sve`, `sve2`, `sme2`. Drives the default instance type. |
| `--instance` | derived from `--isa` | EC2 instance type override, e.g. `c8g.xlarge` |
| `--label` | `f"{dataset}-{isa}"`, else `isa`, else the instance-type tier | Identifies this instance — one per concurrently-desired instance |
| `--dataset` | skip | Build this dataset's native lib (ncnn/llama.cpp) right after provisioning |
| `--initial-build` | skip | Run `make <target>` after provisioning a *fresh* instance only |
| `--on-demand` | off | Provision on-demand instead of spot — won't be reclaimed mid-run, at a higher hourly price |
| `--teardown` | — | Destroy the instance(s) — all recorded labels if `--label` omitted |
| `--status` | — | Show instance status |

## Results and traces

- `results/<definition>_<dataset>_<model>.json` and `.jsonl` — this run's
  outcome per definition (unless `--no-save`).
- `traces/<...>.json` — the full turn-by-turn `version_history`, only with
  `--save-trace`.
- `bench-trace/solutions/` and `bench-trace/traces/` — the kernel itself and
  its evaluation trace, persisted by the MCP server on every new best
  *during* the run, independent of `--no-save`/`--save-trace`.
- `agent-runs-mcp/` — synced back from the remote instance after the run
  (compiled sources, disassembly, trajectory) via
  `mcp_client.sync_bench_trace_back()`.

## File map

| File | Role |
|---|---|
| `provision.py` | Terraform lifecycle (provision/status/teardown) for Graviton EC2 instances |
| `run_benchmark.py` | CLI entry point; per-definition loop; writes `results/`/`traces/` |
| `evaluator.py` | The agent turn loop itself: system/user prompts, tool-call dispatch, retries, history compression |
| `mcp_client.py` | MCP client bridge to `mcp_app/server.py` — the same server nanobot/Claude Code drive in Path 2 |
| `remote.py` | `InstanceHandle` — SSH connection details for a provisioned instance |
| `eval_config.json` | Shared "what's currently up" record, read/written by both `provision.py` and `skills/launch/` |
