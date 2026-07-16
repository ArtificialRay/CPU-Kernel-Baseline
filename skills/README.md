# Kernel auto-optimizing via agent harness

CPU-Kernel-Baseline provides skills to run its kernel auto-optimizing bench
pipeline (compile/evaluate/disassemble/submit against `mcp_app`'s MCP
server) through modern agent harnesses.

This document covers what's common to every harness: which ones are
supported and where their skill lives, how to start an `mcp_app` session
(`skills/launch/`), and how to sync results back afterward. **How to
configure a specific harness's own MCP-server settings, where to put
`SKILL.md` in that harness's directory layout, and how to start that harness
for a run is covered in that harness's own skill `README.md`** — this
document intentionally doesn't repeat it.

## 1. Supported harnesses

| harness | skill | status |
|---|---|---|
| Nanobot | [`nanobot/nanobot-kernel-session/`](nanobot/nanobot-kernel-session/) (`SKILL.md` + `README.md`) | supported |
| Claude Code | — | not yet implemented |
| Gemini CLI | — | not yet implemented |

Each skill directory's `README.md` is the operator-facing setup guide for
that harness; its `SKILL.md` is the agent-facing optimization workflow. Read
this document first, then that harness's own `README.md`.

## 2. Start an mcp_app session

`launch/` (`skills/launch/`) is a harness-agnostic, self-contained package —
zero imports from `eval/` or `mcp_app/` (see `mcp_app/README.md`'s "Scope
boundary") — that provisions a Graviton instance and starts an `mcp_app`
session on it. It duplicates rather than imports `eval/provision.py`'s
Terraform-driving logic (see `launch/provision.py`'s docstring for why); it
shares the *same* `terraform/` state at the repo root (one set of cloud
instances, not two), but keeps its own independent record of which instance
is currently up (`launch/launch_config.json`) — it won't know about an
instance `eval/provision.py` brought up, or vice versa.

The one-shot path, run from `skills/launch/`:

```bash
python3 launch_session.py launch \
    --isa <sve|sve2> --dataset <ncnn|simd-loop|llama.cpp>
```

This reuses an already-up instance for that `isa` tier if `launch/`
provisioned one earlier and it's still reachable, otherwise provisions a
fresh one (Terraform apply, wait for SSH, rsync the repo, install build
deps, build the dataset's native lib if needed) — then starts an `mcp_app`
session on it and prints an MCP spawn command (stdio) or endpoint (sse).

### `launch` flags

| flag | required? | default | notes |
|---|---|---|---|
| `--isa` | yes | — | one of `neon`, `sve`, `sve2`, `sme2` — pick by target hardware; drives the default instance type |
| `--dataset` | yes | — | one of `ncnn`, `simd-loop`, `llama.cpp` — see the harness skill's own doc for the `baseline_author`/`isa` table |
| `--instance` | no | derived from `--isa` | EC2 instance type override, e.g. `c8g.xlarge` |
| `--fresh` | flag | off | force a new instance even if one's already up for this isa tier |
| `--author` | no | `nanobot` | tags every solution/trace this session writes; also names the session's `run_dir` (`agent-runs-mcp/<author>/`) |
| `--baseline-author` | no | auto-derived from `--dataset` | only pass this to override |
| `--local-repo-dir` | **no** | this checkout's own root (`REPO_ROOT`, computed from where `launch_session.py` itself lives — not your shell's cwd) | your local checkout of this repo, pushed to the instance.  Default is the expected repo directory at local side|
| `--remote-root` | no | `~/arm-bench` | where the repo lives on the instance |
| `--transport` | no | `stdio` | `stdio` (default, try first) or `sse` (fallback for MCP clients that can only take a URL, not a spawn command) |
| `--no-sync` | flag | off | skip the rsync step (repo already up to date on the instance) |
| `--skip-preflight` | flag | off | skip the dataset-build check (only if you already know this exact instance's native library is built — this doesn't cover baseline collection, which happens lazily server-side once the agent starts compiling) |

If you'd rather do the two steps separately (e.g. to provision once and
`prepare-session` against it repeatedly), that composes the same way —
`provision` prints the resulting `host`/`user`/`key_file`, feed those into
`prepare-session`:

```bash
python3 launch_session.py provision --isa <sve|sve2> --dataset <dataset>
python3 launch_session.py prepare-session \
    --host <ip> --user ubuntu --key-file ~/.ssh/id_rsa \
    --dataset <ncnn|simd-loop|llama.cpp> --isa <sve|sve2>
```

### `provision` flags

| flag | required? | default | notes |
|---|---|---|---|
| `--isa` | yes | — | one of `neon`, `sve`, `sve2`, `sme2` — drives the default instance type |
| `--instance` | no | derived from `--isa` | EC2 instance type override, e.g. `c8g.xlarge` |
| `--local-repo-dir` | **no** | this checkout's own root (`REPO_ROOT`) | repo checkout to rsync; override only if driving this from a different checkout |
| `--fresh` | flag | off | force a new instance even if one's already up for this isa tier |
| `--dataset` | no | `""` (skip) | build this dataset's native lib right after provisioning |

### `prepare-session` flags

Unlike `launch`/`provision`, this one is meant to be pointed at an instance
you already have `host`/`user`/`key_file` for (e.g. from `provision`'s
output), so `--local-repo-dir` has no `REPO_ROOT` fallback here — it's
genuinely required unless you pass `--no-sync`.

| flag | required? | default | notes |
|---|---|---|---|
| `--host` | yes | — | reachable IP/hostname of the instance |
| `--user` | no | `ubuntu` | |
| `--key-file` | no | `~/.ssh/id_rsa` | |
| `--dataset` | yes | — | one of `ncnn`, `simd-loop`, `llama.cpp` |
| `--isa` | yes | — | one of `neon`, `sve`, `sve2`, `sme2` |
| `--author` | no | `nanobot` | tags every solution/trace this session writes; also names the session's `run_dir` (`agent-runs-mcp/<author>/`) |
| `--baseline-author` | no | auto-derived from `--dataset` | only pass this to override |
| `--local-repo-dir` | **yes, unless `--no-sync`** | — | your local checkout of this repo, pushed to the instance |
| `--remote-root` | no | `~/arm-bench` | where the repo lives on the instance |
| `--transport` | no | `stdio` | `stdio` (default, try first) or `sse` (fallback) |
| `--no-sync` | flag | off | skip the rsync step (repo already up to date on the instance) — makes `--local-repo-dir` optional |
| `--skip-preflight` | flag | off | skip the dataset-build check |

What you do with the printed spawn command/endpoint — where it goes in your
harness's config, what `tool_timeout`/`enabledTools` settings it needs — is
harness-specific; see that harness's own `README.md` (§1's table).

## 3. After the run: sync results back

Once the agent has `submit`'d everything it was assigned (or you decide to
stop it), pull results back — run this from the same host you ran
`provision`/`prepare-session` from, **after** the run has finished:

```bash
python3 launch_session.py sync-results \
    --host <ip> --user ubuntu --key-file ~/.ssh/id_rsa \
    --author <same --author you used before> \
    --local-results-dir <path to your local checkout>/agent-runs-nanobot
```

### `sync-results` flags

| flag | required? | default | notes |
|---|---|---|---|
| `--host` | yes | — | reachable IP/hostname of the instance |
| `--user` | no | `ubuntu` | |
| `--key-file` | no | `~/.ssh/id_rsa` | |
| `--remote-root` | no | `~/arm-bench` | where the repo lives on the instance |
| `--author` | no | `nanobot` | must match the `--author` the session was launched with |
| `--definition` | no | — (pulls everything this author touched) | sync only this definition's subdirectory |
| `--local-results-dir` | yes | — | where to pull results down to |

## Other `launch/` operations

```bash
python3 launch_session.py status              # what launch/ thinks is up
python3 launch_session.py teardown             # terraform-destroy the instance(s)
```

Neither takes any flags. `teardown` shares Terraform state with
`eval/provision.py --teardown` — it tears down the same physical
instance(s) regardless of which side provisioned it.

Fanning a batch of definitions out across N instances means running N
independent copies of `launch`/`sync-results` today (one pair per instance)
— see the harness skill's own "Optimizing many definitions" section for how
work is batched across definitions within a single session.
