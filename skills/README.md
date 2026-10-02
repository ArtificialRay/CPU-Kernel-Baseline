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
| Claude Code | [`claude-code/claude-code-kernel-session/`](claude-code/claude-code-kernel-session/) (`SKILL.md`) | supported |
| Codex | [`codex/codex-kernel-session/`](codex/codex-kernel-session/) (`SKILL.md` + `config.json.example`) | supported |
| Cline | [`cline/cline-kernel-session/`](cline/cline-kernel-session/) (`SKILL.md` + `config.json.example`) | supported |
| Gemini CLI | — | not yet implemented |

Each skill directory's `SKILL.md` is the agent-facing optimization workflow.
Nanobot also has a `README.md`, the operator-facing guide for wiring it to a
session by hand; read this document first, then that one. Claude Code, Codex
and Cline need no manual wiring: `test_scripts/bench_fleet.py --harness <name>`
starts the session, points the harness at it and hands it the skill (see
`test_scripts/harness_adapters.py`).

## 2. Start an mcp_app session

`launch/` (`skills/launch/`) is a harness-agnostic, self-contained package —
zero Python imports from `eval/` or `mcp_app/` — that provisions a Graviton
instance (or the EC2 Mac, for `sme2`) and starts an `mcp_app` session on it. Provisioning itself (Terraform apply/destroy) is
done by the standalone `provisioning/provision.py` script; `launch/launch_session.py`
invokes it only via subprocess, never imports it. Both sides read/write the
same shared `provisioning/eval_config.json` for "what's currently up" — so an
instance `provisioning/provision.py` brought up is visible to `launch/`, and vice
versa; there's exactly one record of what's running, not two.

Note that **launch**, or **provision** + **prepare-session** used separately, brings up an instance with the target dataset (codebase) built. As long as you copy the spawn command for mcp server to your agent's config, or execute the spawn command, the server starts.

The one-shot path, run from `skills/launch/`:

```bash
python3 launch_session.py launch \
    --isa <neon|sve|sve2|sme2> --dataset <ncnn|simd-loop|llama.cpp|kleidiai>
```

This reuses an already-up instance for that `isa` tier if `launch/`
provisioned one earlier and it's still reachable, otherwise provisions a
fresh one (Terraform apply, wait for SSH, rsync the repo, install build
deps, build the dataset's native lib if needed) — then starts a persistent
`mcp_app` server on it in streamable-http mode, reached through an
SSH local-port-forward (not exposed publicly), and prints the endpoint.

### `launch` flags

| flag | required? | default | notes |
|---|---|---|---|
| `--isa` | yes | — | one of `neon`, `sve`, `sve2`, `sme2` — pick by target hardware; drives the default instance type |
| `--dataset` | yes | — | one of `ncnn`, `simd-loop`, `llama.cpp`, `kleidiai` — see the harness skill's own doc for the `baseline_author`/`isa` table. Repeatable: pass it more than once to serve several datasets over one connection |
| `--instance` | no | derived from `--isa` | EC2 instance type override, e.g. `c8g.2xlarge` |
| `--on-demand` | flag | off | provision on-demand instead of spot; only matters when a fresh instance is provisioned |
| `--author` | no | `f'nanobot-{isa}'` | tags every solution/trace this session writes (`f"{author}_{definition.name}"`); also names the session's `run_dir` (`agent-runs-mcp/<author>/`). isa is always folded into the default so two isa's don't clobber each other's solution files — nothing else disambiguates isa. |
| `--baseline-author` | no | auto-derived from `--dataset` | only pass this to override |
| `--label` | no | `f'{dataset(s)}-{author}'` | name identifying this instance — one per concurrently-desired instance (see `provisioning/provision.py`'s module docstring). Since `--author` already carries isa by default, this alone keeps different isa's on separate instances without extra flags |
| `--local-repo-dir` | **no** | this checkout's own root (`REPO_ROOT`, computed from where `launch_session.py` itself lives — not your shell's cwd) | your local checkout of this repo, pushed to the instance by `prepare_session`'s rsync |
| `--remote-root` | no | `~/arm-bench` | where the repo lives on the instance |
| `--local-port` | no | `8888` | local end of the SSH tunnel; fixed, so a reused MCP client config doesn't need re-editing every relaunch |
| `--remote-port` | no | `8765` | port `mcp_app.server` binds to on the instance; change it to run a second session on the same instance |
| `--no-sync` | flag | off | skip the rsync step (repo already up to date on the instance) |

If you'd rather do the two steps separately (e.g. to provision once and
`prepare-session` against it repeatedly), that composes the same way —
`provision` prints the resulting `host`/`user`/`key_file`, feed those into
`prepare-session`:

```bash
python3 launch_session.py provision --isa <neon|sve|sve2|sme2> --dataset <dataset>
python3 launch_session.py prepare-session \
    --host <ip> --user ubuntu --key-file ~/.ssh/id_rsa \
    --dataset <ncnn|simd-loop|llama.cpp|kleidiai> --isa <neon|sve|sve2|sme2>
```

### `provision` flags

| flag | required? | default | notes |
|---|---|---|---|
| `--isa` | yes | — | one of `neon`, `sve`, `sve2`, `sme2` — drives the default instance type |
| `--instance` | no | derived from `--isa` | EC2 instance type override, e.g. `c8g.2xlarge` |
| `--on-demand` | flag | off | provision on-demand instead of spot |
| `--label` | no | `f'{dataset(s)}-{isa}'` | name identifying this instance. `provision` has no `--author` (it's a bare infra command, not tied to any producer), so unlike `launch` its default doesn't fold author in |
| `--local-repo-dir` | no | *(no effect here)* | accepted for parity with `launch`, but unused by standalone `provision` — `provisioning/provision.py` always rsyncs its own repo checkout during provisioning |
| `--dataset` | no | `""` (skip) | build this dataset's native lib right after provisioning: `ncnn`, `simd-loop` or `llama.cpp` |

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
| `--dataset` | yes | — | one of `ncnn`, `simd-loop`, `llama.cpp`, `kleidiai`; repeatable |
| `--isa` | yes | — | one of `neon`, `sve`, `sve2`, `sme2` |
| `--author` | no | `f'nanobot-{isa}'` | tags every solution/trace this session writes; also names the session's `run_dir` (`agent-runs-mcp/<author>/`). |
| `--baseline-author` | no | auto-derived from `--dataset` | only pass this to override |
| `--local-repo-dir` | **yes, unless `--no-sync`** | — | your local checkout of this repo, pushed to the instance |
| `--remote-root` | no | `~/arm-bench` | where the repo lives on the instance |
| `--local-port` | no | `8888` | local end of the SSH tunnel; fixed, so a reused MCP client config doesn't need re-editing every relaunch |
| `--remote-port` | no | `8765` | port `mcp_app.server` binds to on the instance |
| `--no-sync` | flag | off | skip the rsync step (repo already up to date on the instance) — makes `--local-repo-dir` optional |

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
| `--author` | **yes** | — | must match the `--author` the session was actually launched with. No default here —  guessing a static default would silently sync from the wrong directory |
| `--definition` | no | — (pulls everything this author touched) | sync only this definition's subdirectory |
| `--local-results-dir` | yes | — | where to pull results down to |
| `--sync-bench-trace` | flag | off | also pull `bench-trace/solutions/` and `bench-trace/traces/` back from the instance (merged, nothing deleted locally) |

## Other `launch/` operations

```bash
python3 launch_session.py status                       # what launch/ thinks is up
python3 launch_session.py teardown                     # terraform-destroy every recorded instance
python3 launch_session.py teardown --label <label>     # ... or just one
```

`status` takes no flags. `teardown` shares Terraform state with
`provisioning/provision.py --teardown` — it tears down the same physical
instance(s) regardless of which side provisioned it.

Fanning a batch of definitions out across N instances means running N
independent copies of `launch`/`sync-results` today (one pair per instance)
— see the harness skill's own "Optimizing many definitions" section for how
work is batched across definitions within a single session.
