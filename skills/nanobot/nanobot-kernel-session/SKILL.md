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
`disassemble` MCP server should already be connected and visible to
you.

## Ground rules: 
### ONE DEFINITION AT A TIME
- Never target two definitions in the same turn. `evaluate`/`disassemble`/
   all require an explicit `definition` argument (`evaluate`/
  `disassemble` also require `version`), checked against whoever you last
  `compile()`'d — if it doesn't match, you get back `{"status":
  "DEF_CHECK_FAILED"}` instead of it silently acting on the wrong one.
- If you are targeted to optimize one or more definition in one specific ISA, DO NOT fall back to use another ISA (e.g. `sve2` → `sve`) unless the prompt explicitly allows it.

### NEVER USE OPENMP PARALLELIZATION
- Kernel implementation that use OpenMp will be rejected by the evaluator

### KERNEL ARE ALWAYRS RUN AT REMOTE INSTANCE
- Remote instance has all dependencies installed, check dependency at local helps nothing but consume your budget

### Useful guidelines at optimization
- feel free to call builtin `write` tool to write anything you find it is interesting in the optimize process, e.g. `disassemble` output, `evaluate` logs, or your own notes.
- feel free to call builtin `read` tool to read any resource you wrote in the optimize process
- disassemble is a good friend to inpsect if SIMD really helps improving performance, or if you are not sure why your optimization is not working as expected. It can help you understand the generated assembly code and identify potential bottlenecks or inefficiencies.
- for instruction-level cost when scheduling SVE2/NEON/FP code (or to explain a low `ipc_mean`), read the Arm Software Optimization Guide MCP resource for your target hardware (list resources, then read on demand) — per-instruction latency / throughput / utilized-pipeline tables (§3 — SVE integer/FP, ASIMD, load/store, BF16): `docs/neoverse-v2-swog.md` for **Neoverse V2 = Graviton4** (the default target, use unless told otherwise), or `docs/neoverse-v1-swog.md` for **Neoverse V1 = Graviton3** (only when targeting Graviton3; its costs differ). Large — read the relevant §3.x section on demand, not wholesale.

## SKILL referenced:
**KernelWiki** is a useful skill for you to optimize kernel in GPU. Although you are required to optimize kernel in CPU, you can still apply similar optimization techniques if it is applicable. You can use it as a reference, but you are FORBIDDEN to copy the code from it.

# WORKFLOW

## 1. Establish the starting-point baseline (do this first, per definition)

Before writing any optimized code for a given definition:

1. `list_resources()` — you'll see a `<definition>/reference-scalar-kernel.cpp`
   resource for **every** definition in this dataset, present from the
   start (the unoptimized scalar kernel for each). Pick the definition
   you're working on and find its entry.
2. `read_resource()` it. It `#include`s a per-definition `.h` header — that header is NOT exposed as an MCP resource and you don't
   need to read it: its constants are baked in automatically when you
   `compile()`. Don't try `list_resources()`/`read_resource()` on it — it
   will 404.
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
2. `evaluate({})` — correctness + timing + cycle speedup in one call. It also auto-persists the best-performing version the moment it beats your previous best this session — that result is already saved to `trajectory.jsonl` and `bench-trace`.
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