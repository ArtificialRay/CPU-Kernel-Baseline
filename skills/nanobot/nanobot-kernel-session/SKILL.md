---
name: nanobot-kernel-session
description: >
  Drive a CPU-Kernel-Baseline AArch64 SIMD-kernel optimization session
  against a provisioned Graviton instance, via compile/evaluate/disassemble/submit
  MCP tools exposed by mcp_app's server. Use when the user asks to optimize
  an ncnn/simd-loop/llama.cpp kernel definition, run a CPU-Kernel-Baseline
  benchmark session, or mentions CPU-Kernel-Baseline / arm-bench / Graviton
  kernel optimization.
metadata:
  nanobot:
    requires:
      bins: ["ssh", "rsync", "python3"]
      env: []
    always: false
---

# nanobot-kernel-session

Drives one or more kernel-optimization sessions against
`mcp_app`'s MCP server (see `mcp_app/README.md` in the target repo for the
full design). This document is a first draft — validate the `description`
trigger and the exact CLI/config surface against real nanobot behavior
before relying on it (see `mcp_app/README.md`'s Open Items).

## 1. Get a session running

You'll be given (or need to ask the user for): a reachable SSH host/user/key
for an already-provisioned Graviton instance, which `isa` it has
(`sve` = Graviton3, `sve2` = Graviton4), and which definition + dataset to
optimize. Instance provisioning is **not** this skill's job — assume the
instance already exists.

Run, from this skill's `scripts/` directory:

```bash
python3 launch_session.py prepare-session \
    --host <ip> --user ubuntu --key-file ~/.ssh/id_rsa \
    --dataset <ncnn|simd-loop|llama.cpp> --definition <definition-name> \
    --baseline-author <see table below> --isa <sve|sve2> \
    --local-repo-dir <path to your local CPU-Kernel-Baseline checkout>
```

This prints an SSH spawn command. Configure it as your own MCP server (the
same `command`/`args` shape this repo's own `.mcp.json` uses for its
`codegraph` server) and connect to it — you'll then see four tools:
`compile`, `evaluate`, `disassemble`, `submit` (no `read_code` — read
previously-written files via **MCP Resources** instead, listed by
`list_resources()`/read via `read_resource()`).

## 2. Establish the starting-point baseline (do this first)

Before writing any optimized code:

1. `list_resources()` — you'll see a `reference-scalar-kernel.cpp` resource
   present from the start (the unoptimized scalar kernel).
2. `read_resource()` it.
3. `compile({"code": <that content>})` — this becomes version `v1`.
4. `evaluate({"measure": true})`.

Record `v1`'s `time_speedup_geomean`/`cycle_speedup_geomean` — you'll need
them at the end (step 5) to report how much you improved over the naive
starting point, not just over the competitive baseline.

## 3. Optimize

Standard loop:

1. `compile({"code": ...})` your optimized attempt.
2. `evaluate({"measure": false})` — fast correctness check only.
3. `evaluate({"measure": true})` — collect timing + cycle speedup.
4. `disassemble({})` when IPC is low or speedup is unexpectedly poor
   (defaults to your kernel's own symbol).
5. Iterate: compile → evaluate → improve. Use `list_resources()`/
   `read_resource()` to re-read any of your own earlier versions
   (`v2.cpp`, `v3.cpp`, ...) if you need to compare against them.

Metrics from `evaluate({"measure": true})`:
- `time_speedup_geomean` — wall-time speedup vs. the competitive baseline
  (geomean across workloads; >1.0 = faster).
- `cycle_speedup_geomean` — cycle-count speedup vs. the same baseline.
- `ipc_mean` — mean instructions-per-cycle.
- `cache_misses_mean` — mean LLC misses.

## 4. Valid `--dataset` / `--baseline-author` / `--isa` values

Hand-maintained (small, rarely changes — kept in sync by hand rather than
generated, see `mcp_app/README.md`):

| dataset      | baseline_author         | isa (pick by target hardware)          |
|--------------|--------------------------|-----------------------------------------|
| `ncnn`       | `baseline-ncnn-arm`      | `sve` (Graviton3) / `sve2` (Graviton4) |
| `simd-loop`  | `reference`              | `sve` (Graviton3) / `sve2` (Graviton4) |
| `llama.cpp`  | `baseline-llamacpp-arm`  | `sve` (Graviton3) / `sve2` (Graviton4) |

## 5. Finish and report

Before calling `submit`, compare your best version's
`time_speedup_geomean`/`cycle_speedup_geomean` against `v1`'s numbers from
step 2. Call `submit({"explanation": ...})` with **both** numbers in the
explanation — e.g. "vs baseline-ncnn-arm: 1.85x; vs unoptimized
reference-scalar starting point (v1): 5.9x". Both end up recorded in
`trajectory.jsonl`'s final `submit` turn.

After `submit` returns `PASSED` (or you decide to stop), sync results back:

```bash
python3 launch_session.py sync-results \
    --host <ip> --user ubuntu --key-file ~/.ssh/id_rsa \
    --definition <definition-name> \
    --local-results-dir <path to your local checkout>/agent-runs-nanobot
```

## 6. Optimizing many definitions in parallel

Given a list of target definitions (and, for true multi-instance
parallelism, one reachable instance per definition): spawn one subagent per
definition. Each subagent independently repeats steps 1–5 above for its own
`(host, definition)` pair — `launch_session.py` has no shared state between
invocations, so this requires no coordination between subagents beyond the
list of (instance, definition) assignments you hand out up front.

## 7. If `prepare-session`'s stdio spawn command doesn't work

If your MCP client config can only take a URL (not a spawn command), fall
back to `--transport sse` on `prepare-session` — it starts an SSH tunnel and
prints an `http://127.0.0.1:<port>/sse` endpoint instead. This path is less
tested (see `mcp_app/README.md`'s Open Items) — try stdio first.
