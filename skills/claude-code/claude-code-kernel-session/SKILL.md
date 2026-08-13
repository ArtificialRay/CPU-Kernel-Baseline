---
name: claude-code-kernel-session
description: >
  Drive a CPU-Kernel-Baseline AArch64 SIMD-kernel optimization session
  against a provisioned Graviton instance over mcp_app's MCP server. Use
  when asked to optimize an ncnn/simd-loop/llama.cpp kernel definition.
---

# claude-code-kernel-session

Drives one or more kernel-optimization sessions against `mcp_app`'s MCP
server. By the time you're reading this, a `compile`/`evaluate`/
`disassemble`/`submit` MCP server should already be connected and visible
to you — exposed as `mcp__cpu-kernel-baseline__*` tools (you don't need to
type that prefix; your tool list already resolves the real names).

## Ground rules

### ONE DEFINITION AT A TIME
- Never target two definitions in the same turn. `evaluate`/`disassemble`
  both require an explicit `definition` argument (`disassemble` also
  requires `version`), checked against whoever you last `compile()`'d — if
  it doesn't match, you get back `{"status": "DEF_CHECK_FAILED"}` instead
  of it silently acting on the wrong one.
- If you are targeted to optimize a definition in one specific ISA, do NOT
  fall back to another ISA (e.g. `sve2` -> `sve`) unless explicitly told
  you can.

### NEVER USE OPENMP PARALLELIZATION
- Kernel implementations that use OpenMP will be rejected by the evaluator.

### KERNELS ALWAYS RUN ON THE REMOTE INSTANCE
- The remote instance has all dependencies installed already; checking
  dependencies locally helps nothing but consumes your budget.

### Useful guidelines while optimizing
- Feel free to use the Write tool to record anything interesting you find
  during the process — e.g. `disassemble` output, `evaluate` logs, or your
  own notes.
- Feel free to use the Read tool to re-read anything you wrote earlier.
- `disassemble` is a good friend for checking whether SIMD is really
  helping, or for understanding why an optimization isn't working as
  expected — it shows you the generated assembly and can help you spot
  bottlenecks or inefficiencies.

# WORKFLOW

## 1. Establish the starting-point baseline (do this first, per definition)

Before writing any optimized code for a given definition:

1. List the MCP resources exposed by the server — you'll see a
   `<definition>/reference-scalar-kernel.cpp` resource for **every**
   definition in this dataset, present from the start (the unoptimized
   scalar kernel for each). Find the entry for the definition you're
   working on.
2. Read it.
3. `compile({"definition": "<that definition's name>", "code": <that
   content>})` — this becomes version `v1` for that definition.
4. `evaluate({})` — one call, always returns both correctness and
   performance together; there's no separate "measure" flag or faster
   correctness-only mode — the underlying evaluator always runs the full
   timed pass once correctness passes, so there's nothing cheaper to opt
   into.

Record `v1`'s `time_speedup_geomean`/`cycle_speedup_geomean` — you'll need
them at the end to report how much you improved over the naive starting
point, not just over the competitive baseline. If you already have these
numbers from resources written before this run, you can skip this step.

## 2. Optimize

Standard loop for the definition you're currently working on:

1. `compile({"definition": "<same definition>", "code": ...})` your
   optimized attempt.
2. `evaluate({})` — correctness + timing + cycle speedup in one call. It
   also auto-persists the best-performing version the moment it beats your
   previous best this session — that result is already saved to
   `trajectory.jsonl` and `bench-trace`.
3. `disassemble({})` when IPC is low or speedup is unexpectedly poor
   (defaults to your kernel's own symbol).
4. Iterate: compile -> evaluate -> improve. Re-list/re-read your own
   earlier versions or the optimization trajectory instead of
   re-evaluating something you already scored. The resources you can read
   are:
   - `<definition>/vN.cpp` — the kernel source for version N, written by
     `compile()`. Re-read an earlier version to compare against or revert
     to.
   - `<definition>/vN.s` — the disassembled AArch64 for version N, written
     by `disassemble()`. Only exists for versions you actually
     disassembled.
   - `<definition>/trajectory.jsonl` — the full turn-by-turn history for
     this definition (every `compile`/`evaluate`/`disassemble`/`submit`
     call, updated live as you go). Read it to check a past attempt's
     recorded scores instead of re-evaluating it.

### Metrics from evaluate({}) (on `"status": "PASSED"`)
- `max_absolute_error`/`max_relative_error` — correctness, always present.
- `time_speedup_geomean` — wall-time speedup vs. the competitive baseline
  (geomean across workloads; >1.0 = faster).
- `cycle_speedup_geomean` — cycle-count speedup vs. the same baseline.
- `ipc_mean` — mean instructions-per-cycle.
- `cache_misses_mean` — mean LLC misses.
On a non-`PASSED` status, `failed_workload`/`log` say which workload failed
and why (correctness or a runtime/timeout error).

## 3. Finish and report

Once you've decided to stop — see your task message for this run's
iteration floor and any other stopping criteria — call
`submit({"definition": "<definition>"})` for that definition, then stop.
Don't call `submit` for a definition you haven't compiled and evaluated at
least once.
