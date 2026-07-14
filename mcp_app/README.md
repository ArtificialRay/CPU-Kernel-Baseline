# mcp_app — MCP server for kernel-optimization sessions

Exposes `compile`/`evaluate`/`disassemble`/`submit` (the same tool surface
`eval/agent_tools/` gives the SSH-based agent loop) as a proper MCP server,so an external agent harness (e.g. nanobot, claude code) can drive kernel-optimization
sessions.

## Layout

```
mcp_app/
    agent_tools/       # self-contained fork of eval/agent_tools/'s logic — no
                        #   SSH abstraction (nothing to abstract, execution is
                        #   always local to whatever machine runs this code)
        isa.py          # march_for_isa(isa) [explicit table] + verify_isa_available(isa)
                         #   [runtime /proc/cpuinfo safety check, not the source of truth]
        ops.py          # compile_kernel/evaluate_kernel/disassemble_so — adapted from
                         #   eval/agent_tools/remote_runner.py's cmd_* functions
        trajectory.py    # TrajectoryWriter — copied verbatim (pure file I/O)
        base.py          # KernelSession ABC — compile/evaluate/disassemble/submit,
                         #   multi-definition (self._definitions keyed by definition name)
        baseline_readiness.py  # per-definition baseline check-then-collect, called
                                #   lazily from KernelSession.compile() on first touch
        ncnn.py, simd_loop.py, llama_cpp.py   # per-dataset KernelSession subclasses
        registry.py       # resolve_tools(dataset) -> Type[KernelSession]
    session.py         # SessionConfig + build_tools() — server-side bootstrap
    server.py           # the MCP server process itself (--transport stdio|sse)
    resources.py         # MCP Resources over a session's run_dir (read_code's replacement)
    dataset_builds.json    # literal copy of eval/dataset_builds.json (ncnn/llama.cpp native-lib build steps)
    smoke_test_driver.py    # sequential, non-nanobot fallback/testing driver — general dataset/
                             #   definition selection AND the fixed 2-definition verification sweep
                             #   (formerly two files, merged — see the file's own docstring)
    scripts/
        _local_ssh.py        # smoke_test_driver.py's own minimal SSH/rsync (NOT shared with skills/)
        test_mcp_client.py    # plain MCP client, used by smoke_test_driver.py and standalone

skills/nanobot/nanobot-kernel-session/    # sibling to mcp_app/, not nested inside it
    SKILL.md            # agent-facing optimization workflow only
    README.md            # operator-facing: launching a session, syncing results back
    scripts/
        remote.py            # RemoteTarget — the skill's own SSH/rsync (independent of _local_ssh.py)
        dataset_builds.json   # own independent copy, same content as mcp_app/dataset_builds.json
        launch_session.py      # prepare_session()/sync_results() — runs on nanobot's host
```

## Session model

One fresh `mcp_app.server` process per (instance, dataset) — **not** per
definition. `dataset` still picks the process's `KernelSession` subclass
(different harness-lifting logic and toolchain deps per dataset: ncnn needs
`libncnn.a`, llama.cpp needs `libggml.a`), but `compile()` takes `definition`
as a per-call argument, so one process can compile/evaluate/submit every
definition in that dataset across the life of one connection, with no
restart in between. `KernelSession` tracks per-definition state (trajectory
writer, turn counter, last/best compile) keyed by definition name — nothing
is evicted, so switching definitions and coming back later still works
correctly (see `agent_tools/base.py`).

This is still what lets a parallel dispatcher (nanobot spawning N subagents
across N instances) fan sessions out without any shared-state coordination
inside `mcp_app` — one subagent per (instance, dataset) pair, each free to
walk as many or as few definitions of that dataset as it's assigned.

## Transport

Two modes, in preference order — try the simpler one first:

1. **stdio-over-ssh (default/preferred)**: the MCP client (nanobot, or
   `test_mcp_client.py`) spawns `ssh user@host "cd <remote_root> && python3 -m
   mcp_app.server --transport stdio ..."` directly as its own subprocess. SSH
   transparently carries the stdio pipes — no port, no listener, no tunnel.
2. **SSE-over-SSH-tunnel (fallback)**: only build/exercise this
   (`server.py`'s `--transport sse` branch, `_run_sse` in `server.py`) if
   stdio-over-ssh turns out not to work with nanobot's real MCP config
   format (i.e. its config can only take a URL, not a spawn command). The
   server binds to `127.0.0.1:<port>` only — never a public interface.

## ISA — explicit, never auto-detected from an instance-type table

`isa` (`neon`/`sve`/`sve2`/`sme2`) is a required parameter everywhere
(`--isa` on `server.py`/`smoke_test_driver.py`, `isa=` on
`launch_session.prepare_session()`) because reproducible experiments need a
known, visible "which SIMD ISA am I testing" — not something inferred
silently from whatever hardware happens to be attached.

- `mcp_app/agent_tools/isa.py::march_for_isa(isa)` is the *only* place the
  isa→march-flags mapping lives, and it's what decides compile flags.
- `verify_isa_available(isa)` probes `/proc/cpuinfo` at session startup as a
  **safety check** (fail loudly if the box doesn't actually support what was
  requested) — it never decides march flags.
- Which *instance type* satisfies a given isa is a provisioning concern,
  entirely outside `mcp_app` (see Scope boundary above).

## `read_code` is retired

Reading previously-written `vN.cpp`/`vN.s`/`trajectory.jsonl`/
`reference-scalar-kernel.cpp` happens via **MCP Resources**
(`mcp_app/resources.py`) instead of a bespoke tool — works identically
regardless of transport or harness co-location. `compile()`'s `source_file`
and `disassemble()`'s `asm_file` responses still include the absolute path
too, as a convenience when the caller happens to share a filesystem with the
server, but Resources are the protocol-correct read path.

## Establishing a "naive starting point" baseline

`session.py::_write_reference_scalar_kernels()` writes **every** definition
in this dataset's `reference-scalar` baseline solution `kernel.cpp` to
`run_dir/<definition_name>/reference-scalar-kernel.cpp` up front, before the
server starts accepting tool calls — this has to stay eager (unlike baseline
*collection*, see below) because the agent needs to read a definition's
reference kernel via `list_resources()`/`read_resource()` and `compile()` it
as that definition's `v1` *before* its own first real `compile()` call for
that definition; by the time `compile()` returns for a new definition, it's
too late to still be the naive starting point. Cheap even across every
definition in the dataset — pure text-file I/O, no compile/evaluate. The
nanobot skill's workflow (see
`skills/nanobot/nanobot-kernel-session/SKILL.md`) tells the agent to
`compile()` + `evaluate()` this as its own first tool call for each
definition it works on — it becomes `v1` in that definition's trajectory
naturally, giving a measured "how slow was the unoptimized starting point"
data point for free, with no new tool needed. This also directly benefits
the existing `.claude/skills/case-study` skill, which previously only had
the reference-scalar source as text.

## Dataset readiness & baseline collection

Two separate checks, at two different times, since one is dataset-level and
one is definition-level:

- **Dataset-level (native library build)**: ncnn/llama.cpp baselines link
  against a native library (`libncnn.a` / `libggml.a`) that isn't part of
  the synced repo — it has to be separately cloned + cmake-built on the
  target instance (`~/ncnn`, `~/llama.cpp`). This still runs upfront, before
  the server starts (it's needed regardless of which definitions get
  touched): `ensure_dataset_ready()`, two independent copies, neither
  importing from the other or from `eval/`:
  - **`mcp_app/smoke_test_driver.py::ensure_dataset_ready()`** — runs before
    its batch baseline collection, for smoke-testing.
  - **`skills/nanobot/nanobot-kernel-session/scripts/launch_session.py::ensure_dataset_ready()`**
    — runs automatically inside `prepare_session()` (skip with
    `--skip-preflight`). A real nanobot session never touches
    `smoke_test_driver.py`, which is smoke-test-only.
- **Definition-level (does this definition have a PASSED baseline trace)**:
  moved in-process into `mcp_app` itself —
  **`mcp_app/agent_tools/baseline_readiness.py::ensure_baseline_collected()`**
  — called lazily from `KernelSession.compile()` the first time a given
  definition is touched (not upfront, since the server no longer knows its
  definition set in advance). Runs `bench.benchmark.Benchmark` directly
  against the server's own already-loaded `TraceSet` rather than shelling
  out to `bench.cli collect-baselines`, so the new trace is reflected
  immediately with no reload needed. This is a **third**, independent copy
  of the check-then-collect logic (alongside the two `ensure_dataset_ready()`
  copies above — matches the existing duplication convention). Because this
  can take a while on first touch, the MCP server config entry for this
  connection should set a generous `tool_timeout` (e.g. 300–600s) — see
  `skills/nanobot/nanobot-kernel-session/SKILL.md`.

Skipping this doesn't fail loudly on its own: the baseline solution just
fails to compile, `evaluate()`/`submit()` still return `PASSED` for the
*agent's own kernel* (which never depends on that library — see below), but
`time_speedup`/`cycle_speedup` come back silently `None` since there's no
valid baseline to compare against.

Both `ensure_dataset_ready()` copies read their own literal copy of
`eval/dataset_builds.json`'s content —
**`mcp_app/dataset_builds.json`** and
**`skills/nanobot/nanobot-kernel-session/scripts/dataset_builds.json`** — file
copies, not runtime reads of anything under `eval/`. This is a real
duplication tradeoff (unlike the small 3-row dataset/baseline_author table —
these are several multi-flag cmake invocations, more likely to drift): if
`eval/dataset_builds.json` changes, **both** copies need a matching manual
`cp`. Check all three when touching any..

## Running it

```bash
# On the target instance, once the repo is synced there. --baseline-author
# is optional — auto-derived from --dataset if omitted (see
# agent_tools/baseline_readiness.py::DEFAULT_BASELINE_AUTHOR). One process
# now serves every definition in the dataset — compile() takes `definition`
# as a per-call argument, not a startup flag.
python -m mcp_app.server --dataset ncnn --author test --isa sve2 \
    --run-dir ~/arm-bench/agent-runs-mcp/test --transport stdio
```

## Smoke test

``` bash
# From wherever, sequential batch testing without nanobot (also does the
# dataset-readiness + baseline-collection preflight before each session):
python -m mcp_app.smoke_test_driver --host 1.2.3.4 --user ubuntu --key-file ~/.ssh/id_rsa \
    --dataset ncnn --baseline-author baseline-ncnn-arm --isa sve2 --all

# End-to-end verification against this repo's 2 standard definitions
# (see Verification below) — same tool, just --problem scoped to one each:
python -m mcp_app.smoke_test_driver --host 1.2.3.4 --user ubuntu --key-file ~/.ssh/id_rsa \
    --dataset ncnn --baseline-author baseline-ncnn-arm --isa sve2 \
    --problem conv2d_fp32_kh1_kw1_sh1_sw1_dh1_dw1_p0
python -m mcp_app.smoke_test_driver --host 1.2.3.4 --user ubuntu --key-file ~/.ssh/id_rsa \
    --dataset llama.cpp --baseline-author baseline-llamacpp-arm --isa sve2 \
    --problem gemm_bf16_n1024_k2048
```

## Verification

Two of the definitions confirmed (at design time) to already have baseline
solutions in `bench-trace/`:

- **ncnn**: `conv2d_fp32_kh1_kw1_sh1_sw1_dh1_dw1_p0`
- **llama.cpp**: `gemm_bf16_n1024_k2048`

**Step 1 — real Graviton instance, stdio-over-ssh, no nanobot.** Provision
an `sve2`-tier instance beforehand (`python eval/provision.py --isa sve2`,
run separately — not something `mcp_app` invokes), then run
`smoke_test_driver.py` against each of the two definitions above (see the
`python -m mcp_app.smoke_test_driver ...` examples in "Running it").

**Step 2 — real nanobot integration attempt, stdio-over-ssh.** Feed
`launch_session.prepare_session(..., transport="stdio")`'s output directly
into nanobot's own MCP config and see if it can drive a real session end to
end. If it works, that's the shipped path — stop here.

**Step 3 — SSE-over-tunnel fallback**, only if Step 2 shows nanobot's config
can't take a spawn command. Not built/exercised until then.

This repo's sandbox (used to write this code) has no ARM/clang toolchain and
no Graviton hardware — `compile()`/`evaluate()` cannot be exercised
end-to-end here. What *was* verified here: all modules import cleanly, and
the MCP tool/resource plumbing (`list_tools`/`call_tool`/`list_resources`/
`read_resource`) responds correctly via stdio transport, including the
expected `COMPILE_ERROR` path when `clang++`/`llvm-objdump` aren't on PATH.
Real compile/evaluate verification (Steps 1–3 above) needs to run on an
actual Graviton instance.

## Open items

1. Exact `/proc/cpuinfo` `Features` tokens distinguishing Graviton3 (sve) vs.
   Graviton4 (sve2) — `isa.py`'s `_ISA_CPUINFO_TOKENS` is a best guess
   (`"sve"`/`"sve2"`/`"asimd"`/`"sme2"`); confirm against real hardware. Only
   affects the precision of `verify_isa_available`'s safety check, never
   `march_for_isa`'s compiled output.
2. Exact `mcp` SDK SSE/streamable-HTTP behavior — `server.py::_run_sse` and
   `test_mcp_client.py::run_sse_sequence` are written against the installed
   `mcp==1.28.1` API surface but untested end-to-end (no client to test
   against yet) — only matters if Step 3 above gets exercised.
3. Whatever provisions/tracks multiple instances for future parallel
   dispatch is entirely outside `mcp_app`'s scope (see Scope boundary) —
   nothing here conflicts with it, but it isn't built.
4. `skills/nanobot/nanobot-kernel-session/SKILL.md` is a first draft in
   nanobot's documented format — validate `description` triggering and
   `metadata.nanobot` fields against real nanobot behavior once testable.
5. Baseline auto-collection (`smoke_test_driver.py::ensure_baselines`) is per-instance
   by construction (each instance's `bench-trace/` is separate) — a future
   multi-instance parallel dispatcher must call it independently per
   instance, and should run those independent calls concurrently too, not
   sequentially instance-by-instance.
6. `mcp_app/dataset_builds.json` and
   `skills/nanobot/nanobot-kernel-session/scripts/dataset_builds.json` are
   both manually-synced copies of `eval/dataset_builds.json` — no automated
   drift check exists. If you edit one, check all three.
7. `mcp_app/smoke_test_driver.py::run_definition()` still spawns one fresh
   server process per definition (compatibility-only update when `compile()`
   gained its `definition` argument — see git history). Collapsing it to one
   server process per dataset run, reusing a single MCP session across all
   of `problem_defs`, is a real efficiency win but is a separate refactor of
   this file's control flow, not done yet.
8. No `list_definitions` discovery tool/resource exists yet — an agent still
   has to be told externally which definition names to `compile()` for a
   given dataset (e.g. by whoever drives nanobot), rather than being able to
   enumerate them itself from the live session. Additive, independent of
   everything else here; a likely fast-follow.
