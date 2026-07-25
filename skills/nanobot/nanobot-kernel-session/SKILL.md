---
name: nanobot-kernel-session
description: >
  Drive a CPU-Kernel-Baseline AArch64 SIMD-kernel optimization session
  against a provisioned Graviton instance. Use when the user asks to optimize
  an ncnn/simd-loop/llama.cpp kernel definition
metadata:
  nanobot:
    requires:
      bins: []
      env: []
    always: true
---

# nanobot-kernel-session

Drives one or more kernel-optimization sessions against `mcp_app`'s MCP
server
By the time you're reading this as the driving agent, a `compile`/`evaluate`/
`disassemble`/`submit` MCP server should already be connected and visible to
you.

## Ground rules: 
### ONE DEFINITION AT A TIME
- Never target two definitions in the same turn. `evaluate`/`disassemble`/
  `submit` take no `definition` arg — they act on whoever you last `compile()`'d.
- If you are targeted to optimize one or more definition in one specific ISA, DO NOT fall back to use another ISA (e.g. `sve2` → `sve`) unless the prompt explicitly allows it.

### KERNEL ARE ALWAYRS RUN AT REMOTE INSTANCE
- Remote instance has all dependencies installed, check dependency at local helps nothing but consume your budget

### Useful guidelines at optimization
- feel free to call builtin `write` tool to write anything you find it is interesting in the optimize process, e.g. `disassemble` output, `evaluate` logs, or your own notes.
- feel free to call builtin `read` tool to read any resource you wrote in the optimize process
- disassemble is a good friend to inpsect if SIMD really helps improving performance, or if you are not sure why your optimization is not working as expected. It can help you understand the generated assembly code and identify potential bottlenecks or inefficiencies.

## SKILL referenced:
**KernelWiki** is a useful skill for you to optimize kernel in GPU. Although you are required to optimize kernel in CPU, you can still apply similar optimization techniques if it is applicable. You can use it as a reference, but you are FORBIDDEN to copy the code from it.

# WORKFLOW

## 1. Establish the starting-point baseline (do this first, per definition)

Before writing any optimized code for a given definition:

1. `list_resources()` — you'll see a `<definition>/reference-scalar-kernel.cpp`
   resource for **every** definition in this dataset, present from the
   start (the unoptimized scalar kernel for each). Pick the definition
   you're working on and find its entry.
2. `read_resource()` it.
3. `compile({"definition": "<that definition's name>", "code": <that content>})`
   — this becomes version `v1` for that definition.
4. `evaluate({})` — one call, always returns both correctness and
   performance together (see §2's Metrics list; there's no separate
   "measure" flag or faster correctness-only mode — the underlying evaluator
   always runs the full timed pass once correctness passes, so there's
   nothing cheaper to opt into).

Record `v1`'s `time_speedup_geomean`/`cycle_speedup_geomean` — you'll need
them at the end (§4) to report how much you improved over the naive
starting point, not just over the competitive baseline; If you already gain the speedup numbers from resources before optimization, you can skip this step.

## 2. Optimize

Standard loop for the definition you're currently working on:

1. `compile({"definition": "<same definition>", "code": ...})` your optimized attempt.
2. `evaluate({})` — correctness + timing + cycle speedup in one call. It also auto-persists the best-performing version the moment it beats your previous best this session — that result is already saved to `trajectory.jsonl` and `bench-trace` even before you call `submit`; see §3.
3. `disassemble({})` when IPC is low or speedup is unexpectedly poor
   (defaults to your kernel's own symbol).
4. Iterate: compile → evaluate → improve. Use `list_resources()`/
   `read_resource()` to re-read any of your own earlier versions or optimization trajectory. The resources you can read are:
   - `<definition>/vN.cpp` — the kernel source for version N, written by `compile()`. Re-read an earlier version to compare against or revert to.
   - `<definition>/vN.s` — the disassembled AArch64 for version N, written by `disassemble()`. Only exists for versions you actually disassembled.
   - `<definition>/trajectory.jsonl` — the full turn-by-turn history for this definition (every `compile`/`evaluate`/`disassemble`/`submit` call, updated live as you go). Read it to check a past attempt's recorded scores instead of re-evaluating it.

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

Before calling `submit` for a definition, compare its best version's
`time_speedup_geomean`/`cycle_speedup_geomean` against that definition's
`v1` numbers from §1. Call `submit({"explanation": ...})` with **both**
numbers in the explanation — e.g. "vs baseline-ncnn-arm: 1.85x; vs
unoptimized reference-scalar starting point (v1): 5.9x". Both end up
recorded in that definition's `trajectory.jsonl`'s final `submit` turn.

Once you've `submit`'d every definition you were assigned (or you decide to
stop), results get synced back by whoever is orchestrating this session.

## Recovering from an MCP session reset

Don't restart from `reference-scalar-kernel.cpp` — nothing is actually lost.
`compile()`/`disassemble()` write every version's source/asm straight to
disk under that definition's run-dir as they happen, and `evaluate()`
auto-persists a `submit` turn the moment it finds a new best (§2) — all of
that survives a reset independent of the live session:

1. `list_resources()` → find that definition's `vN.cpp` entries and `trajectory.jsonl`.
2. `read_resource()` the `trajectory.jsonl` and find the **last `submit` turn**
   — its `source_file` names the actual best version (not necessarily the
   highest `vN.cpp`, if a later attempt regressed) and its `metrics` carries
   the recorded speedups, so there's no need to scan/compare earlier turns.
3. `read_resource()` that `vN.cpp` and `compile()` it again to resume — it
   becomes a new `v1`, same working code, just a fresh version count.
4. For the final §3 report, keep the *original* `v1` numbers from
   `trajectory.jsonl`'s first `compile`/`evaluate` turns — not the numbers
   from this recovery compile.