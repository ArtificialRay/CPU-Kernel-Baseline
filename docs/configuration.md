# Configuration

There are a lot of config files, but they fall into three groups and only the
first one is something you have to set up yourself.

## 1. Set up once (git-ignored — copy each from the `.example` next to it)

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

## 2. Checked-in defaults (edit only to change how kernels are evaluated or built)

| File | Purpose |
|---|---|
| `config/kernel_contracts.yaml` | Kernel evaluation parameters: op-type correctness/timing overrides, disallowed source patterns, ISA→march mapping, baseline authors |
| `config/dataset_builds.json` | Step-by-step clone/build of each dataset's native lib (ncnn, ggml) on a remote instance, plus the pinned llama.cpp tools tree for the end-to-end runs |
| `config/e2e_models.json` | Models for the end-to-end tokens/s runs (`scripts/e2e/`) |
| `config/rsync_allowlist.json` | Repo paths synced to instances before a session (override once with `RSYNC_ALLOWLIST` in `.env`) |

## 3. Written by the tools (do not edit by hand)

| File | Purpose |
|---|---|
| `provisioning/eval_config.json` | The instances currently up (host, user, key). Created and updated by `provision.py`; `eval_config.json.example` only shows the format. |
| `terraform/terraform.tfstate.d/` | Local Terraform state, one directory per workspace |
| `agent-runs-<author>/`, `harness_trajs/` | Per-kernel trajectories and logs synced back from the instances |

## Notes on remote instances

- **Python environment:** every instance (Graviton and Mac alike) runs `bench/`
  and `mcp_app/` from one uv-managed venv at `~/venv`, built from
  `requirements.txt` by `provision.py`. Nothing to set up by hand, and a reused
  instance is checked (and repaired if a package is missing) before a run.
- **Linux boxes that `bench_fleet.py` drives shut themselves down when it stops.**
  It keeps a shutdown scheduled `--watchdog-minutes` (default 120) ahead while
  baselines are collected and during every job.
  Terraform turns that shutdown into a termination, so a box left behind by a
  crashed driver or a closed laptop stops billing within about 2 hours. A box
  you keep around between runs is reprovisioned on the next run if the
  watchdog took it. `--watchdog-minutes 0` cancels it. Mac instances are not
  watched: their Dedicated Host bills either way.
- **Mac (`mac-m4.metal`) needs a Dedicated Host.** Provisioning picks an
  existing `available` host with no instance on it and never allocates one
  itself (AWS bills a 24-hour minimum). Allocate one first:
  `aws ec2 allocate-hosts --instance-type mac-m4.metal --availability-zone <az> --quantity 1`.
  A host that just lost its instance stays `pending` for a while while AWS
  scrubs it.

## Optional environment variables

Only needed in specific cases, set in `.env` or the shell.

| Variable | Effect |
|---|---|
| `NCNN_ROOT`, `LLAMA_CPP_ROOT` | Point the local `bench/` harness at an existing ncnn / llama.cpp checkout |
| `ARMBENCH_NOTEPAD=1` | Give the `own` harness's agent a persistent scratchpad tool |
| `ARMBENCH_CC_JOB_ROOT` | Parent of the per-job working directories `--harness claude-code` creates (default: the system temp dir); must be outside this repo |
| `OPENROUTER_API_KEY` | Used by `scripts/bench_loop_agent.py` |
| `WANDB_INSTANCE_TYPE` | Label recorded on runs logged with `--wandb` |

## Timing protocol

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

## nanobot model and provider

`--harness nanobot` reads its base config from
`skills/nanobot/nanobot-kernel-session/config.json` by default. Passing
`--model` alone only overrides `agents.defaults.model` — the provider (and
its API key) still comes from that checked-in config, so switching to a
model from a different provider needs its own base config. Set
`NANOBOT_CONFIG_BASE` in `.env` (see `.env.example`) to point at one instead.
