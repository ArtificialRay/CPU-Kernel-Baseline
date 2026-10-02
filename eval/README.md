# eval/ — own harness (in-repo litellm agent loop) and single-shot

Two ways to run a model against the MCP server without an external agent CLI.
Both are driven by `test_scripts/bench_fleet.py`, which owns the same
lifecycle it gives every other harness: provision or reuse the instance,
start `mcp_app/server.py` on it, run the jobs, sync results back.

- `--harness own` — `eval/evaluator.py::run_agentic_eval`, a litellm tool-call
  loop. The model calls `check_progress` / `compile` / `evaluate` /
  `disassemble` until it stops or runs out of budget. There is no separate
  submit tool here: `evaluate()` persists the best version on the instance.
- `--harness single-shot` — `eval/single_shot.py::run_single_shot`. One
  completion with no tools, `--samples` times per definition. Each kernel that
  comes back is compiled and measured once through the same MCP
  `compile`/`evaluate`, so the number is comparable with a multi-turn run.

## Prerequisites

```bash
pip install -r requirements.txt   # from repo root
```

- The kernel dataset in `bench-trace/` at the repo root (see the top-level
  README).
- Model credentials: either the provider's usual environment variable
  (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, ..., from the
  shell or a `.env` at the repo root), or `eval/llm_providers.json` (copy
  `llm_providers.json.example`), which sets `api_key` / `api_base` per
  provider and falls back to the environment for anything left out.
- For provisioning: an AWS account, Terraform (`terraform/`), an SSH key and
  `provisioning/workspaces.json` (see the top-level README's Configuration
  section). `provisioning/eval_config.json` is written by the tools and
  records the instances that are up, one per label.

## Quickstart

```bash
python3 test_scripts/bench_fleet.py --harness own \
    --dataset ncnn --isa sve --model anthropic/claude-opus-4-8
```

What this does, end to end:
1. Provisions a fresh instance, or reuses a reachable one recorded in
   `provisioning/eval_config.json` under this run's label, and collects any
   missing baseline traces on it.
2. Starts `mcp_app.server` there and opens one MCP client session
   (`eval/mcp_client.py::attach()`), shared by every definition in the run.
3. Runs the loop for each definition matching `--dataset` (narrow with
   `--definitions`). The MCP server rejects further compile/evaluate/
   disassemble calls after `--max-iterations` of them.
4. Syncs each definition's run directory back and writes the result JSON
   (see "Results" below).
5. Tears down the instance the run used.

## Usage examples

**One definition:**
```bash
python3 test_scripts/bench_fleet.py --harness own --model anthropic/claude-opus-4-8 \
    --dataset ncnn --isa sve --definitions "conv2d_fp32_kh3_kw3_sh1_sw1_dh1_dw1_p1"
```

**Another dataset or ISA** (`--dataset`: `ncnn`, `simd-loop`, `llama.cpp`,
`kleidiai`; `--isa`: `neon`, `sve`, `sve2`, `sme2`, `portable`):
```bash
python3 test_scripts/bench_fleet.py --harness own --model anthropic/claude-opus-4-8 \
    --dataset simd-loop --isa sve2
```

**The `portable` C/C++-only ablation** (agent code may not use NEON/SVE
intrinsics; compiled with the same flags as `neon`):
```bash
python3 test_scripts/bench_fleet.py --harness own --model anthropic/claude-opus-4-8 \
    --dataset simd-loop --isa portable
```

**Single shot, five samples per definition:**
```bash
python3 test_scripts/bench_fleet.py --harness single-shot --model anthropic/claude-opus-4-8 \
    --dataset llama.cpp --isa sve --samples 5
```

**Keep resuming until every definition has a complete trajectory** (not
available for `single-shot`, which never submits):
```bash
python3 test_scripts/bench_fleet.py --harness own --model anthropic/claude-opus-4-8 \
    --dataset ncnn llama.cpp --isa sve --until-complete
```

## Options that matter here

Run `python3 test_scripts/bench_fleet.py --help` for the full list.

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | (required) | litellm model string, e.g. `anthropic/claude-opus-4-8` |
| `--dataset` | (required) | `ncnn`, `simd-loop`, `llama.cpp` or `kleidiai` |
| `--isa` | `sve` | `neon`, `sve`, `sve2`, `sme2`, `portable` |
| `--definitions` | all | One name, a space-separated list, or a quoted JSON array |
| `--max-iterations` | `40` | Hard cap on compile/evaluate/disassemble calls per definition, enforced by the MCP server and stated in the prompt. The `own` loop itself stops after three times as many turns. |
| `--min-iterations` | `15` | Floor: the model is told not to stop earlier |
| `--samples` | `3` | `single-shot` only: independent generations per definition. Failed samples are kept, so the pass rate is part of the result. |
| `--temperature` | `1.0` | `single-shot` only |
| `--retries` | `3` | Retries for transient infrastructure failures |
| `--on-demand` | off | On-demand instead of spot, when a fresh instance is provisioned |

Two more knobs live outside the command line:

- `ARMBENCH_NOTEPAD=1` gives the `own` loop's agent a `notepad` tool whose
  contents survive history compression.
- `tool_call_loop` in `config/kernel_contracts.yaml` holds the loop's
  temperature, completion timeout, retry budget and MCP client timeouts, plus
  the per-model exceptions (models that reject `temperature`, and models that
  need the Responses API instead of Chat Completions, handled in
  `eval/llm_call.py`).

### Instance types

`--isa` picks the instance type (`isa` table in `config/kernel_contracts.yaml`);
`--instance` overrides it.

| ISA | Instance | Notes |
|-----|----------|-------|
| `neon` | `c7g.xlarge` | Graviton3 |
| `portable` | `c7g.large` | Graviton3; not in the `isa` table, so `bench_fleet.py`'s fallback type applies |
| `sve` | `c7g.xlarge` | Graviton3, Neoverse V1, 256-bit SVE |
| `sve2` | `c8g.xlarge` | Graviton4, Neoverse V2, 128-bit SVE2 |
| `sme2` | `mac-m4.metal` | Apple M4; needs a Dedicated Host (see the top-level README) |

## Provisioning (`provisioning/provision.py`)

Standalone script; `bench_fleet.py` and `skills/launch/launch_session.py` call
into it. Useful directly when you want an instance to persist across several
runs, or to check or tear down what is currently up.

```bash
python provisioning/provision.py --isa sve2                    # provision (label defaults to isa)
python provisioning/provision.py --isa sve2 --dataset ncnn      # + build ncnn's native lib right after
python provisioning/provision.py --status                        # show what's currently up
python provisioning/provision.py --teardown                       # destroy every recorded instance
python provisioning/provision.py --teardown --label ncnn-sve2     # destroy just one label
python provisioning/provision.py --isa sve2 --on-demand            # on-demand, not spot — for long unattended runs
```

| Flag | Default | Description |
|------|---------|-------------|
| `--isa` | — | ISA target: `neon`, `sve`, `sve2`, `sme2`. Drives the default instance type. |
| `--instance` | derived from `--isa` | EC2 instance type override, e.g. `c8g.2xlarge` |
| `--label` | `f"{dataset}-{isa}"`, else `isa`, else the instance-type tier | Identifies this instance — one per concurrently-desired instance |
| `--dataset` | skip | Build this dataset's native lib (ncnn/llama.cpp) right after provisioning |
| `--initial-build` | skip | Run `make <target>` after provisioning a *fresh* instance only |
| `--on-demand` | off | Provision on-demand instead of spot — won't be reclaimed mid-run, at a higher hourly price |
| `--teardown` | — | Destroy the instance(s) — all recorded labels if `--label` omitted |
| `--status` | — | Show instance status |

## Results

`<author>` defaults to `<harness>-<model>-<isa>`, e.g.
`own-claude-opus-4-8-sve`.

- `harness_trajs/<harness>/<author>/<dataset>_<isa>_<definition>.log` — the
  result JSON of that definition. `own`: `status` (`PASSED` with the best
  version of the session, or `NO_SUBMIT`), `time_speedup`, `cycle_speedup`
  and the full `version_history`. `single-shot`: one row per sample plus the
  `pass_rate`.
- `agent-runs-<author>/<definition>/` — synced back from the instance:
  `trajectory.jsonl`, every compiled version (`v<N>.cpp`), disassembly, and
  the reference scalar kernel.
- `bench-trace/solutions/` on the instance holds the persisted kernels; pass
  `--sync-solutions` to pull them back as well.

## File map

| File | Role |
|---|---|
| `evaluator.py` | The agent turn loop: system/user prompts, tool-call dispatch, retries, history compression, optional notepad |
| `single_shot.py` | One completion, no tools; extracts the kernel, then compiles and measures it over MCP |
| `mcp_client.py` | MCP client bridge to `mcp_app/server.py` — the same server the external harnesses drive |
| `llm_call.py` | Chooses Chat Completions or the Responses API per model and normalizes the reply |
| `llm_providers.py` | Optional per-provider `api_key` / `api_base` overrides from `llm_providers.json` |
