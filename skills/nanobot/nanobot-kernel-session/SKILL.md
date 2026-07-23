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

## Ground rules: one definition at a time

- Never target two definitions in the same turn. `evaluate`/`disassemble`/
  `submit` take no `definition` arg — they act on whoever you last `compile()`'d.
- Interleaving compiles across definitions silently drops the earlier one's
  evaluation, and gains nothing — these calls run sequentially regardless.
- Finish one definition's full chain (`compile` → `evaluate` → `disassemble`
  if needed → `submit`) before moving to the next.
- Revisiting an earlier definition later is fine — its version counter and
  `best_compile` pick up where you left off (state is never evicted).

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
starting point, not just over the competitive baseline.

## 2. Optimize

Standard loop for the definition you're currently working on:

1. `compile({"definition": "<same definition>", "code": ...})` your optimized attempt.
2. `evaluate({})` — correctness + timing + cycle speedup in one call.
3. `disassemble({})` when IPC is low or speedup is unexpectedly poor
   (defaults to your kernel's own symbol).
4. Iterate: compile → evaluate → improve. Use `list_resources()`/
   `read_resource()` to re-read any of your own earlier versions
   (`<definition>/v2.cpp`, `<definition>/v3.cpp`, ...) if you need to compare
   against them.

### Metrics from evaluate({}) (on `"status": "PASSED"`)
- `max_absolute_error`/`max_relative_error` — correctness, always present.
- `time_speedup_geomean` — wall-time speedup vs. the competitive baseline
  (geomean across workloads; >1.0 = faster).
- `cycle_speedup_geomean` — cycle-count speedup vs. the same baseline.
- `ipc_mean` — mean instructions-per-cycle.
- `cache_misses_mean` — mean LLC misses.
On a non-`PASSED` status, `failed_workload`/`log` say which workload failed
and why (correctness or a runtime/timeout error).

### Useful guidelines
- feel free to call builtin `write` tool to write anything you find it is interesting in the optimize process, e.g. `disassemble` output, `evaluate` logs, or your own notes.
- feel free to call builtin `read` tool to read any resource you wrote in the optimize process
- disassemble is a good friend to inpsect if SIMD really helps improving performance, or if you are not sure why your optimization is not working as expected. It can help you understand the generated assembly code and identify potential bottlenecks or inefficiencies.


## 3. Valid `--dataset` / `--baseline-author` / `--isa` values

Hand-maintained (small, rarely changes — kept in sync by hand rather than generated, see `mcp_app/README.md`):

| dataset      | baseline_author         | isa (pick by target hardware)          |
|--------------|--------------------------|-----------------------------------------|
| `ncnn`       | `baseline-ncnn-arm`      | `sve` (Graviton3) / `sve2` (Graviton4) |
| `simd-loop`  | `reference`              | `sve` (Graviton3) / `sve2` (Graviton4) |
| `llama.cpp`  | `baseline-llamacpp-arm`  | `sve` (Graviton3) / `sve2` (Graviton4) |

`baseline_author` is only needed here if you want to override the server's
auto-derived default — you no longer have to pass it explicitly.

`portable` is also a valid `--isa` for any dataset: it forces the baseline
`-march=armv8-a` and means **write portable C++ only — no NEON/SVE intrinsics**
(the compiler may still auto-vectorize). Use it for the without-SIMD ablation.

## 4. Finish and report

Before calling `submit` for a definition, compare its best version's
`time_speedup_geomean`/`cycle_speedup_geomean` against that definition's
`v1` numbers from §1. Call `submit({"explanation": ...})` with **both**
numbers in the explanation — e.g. "vs baseline-ncnn-arm: 1.85x; vs
unoptimized reference-scalar starting point (v1): 5.9x". Both end up
recorded in that definition's `trajectory.jsonl`'s final `submit` turn.

Once you've `submit`'d every definition you were assigned (or you decide to
stop), results get synced back by whoever is orchestrating this session.

## Recovering from an MCP session reset

If you find the server/the session was reset, please
**Don't just restart from `reference-scalar-kernel.cpp`.** Every
`compile()`/`disassemble()` call writes its source/asm straight to disk
under that definition's run-dir as it happens (`v1.cpp`, `v2.cpp`, ...,
`trajectory.jsonl`) — this is independent of the in-memory session, so it
survives a reset even though the live session doesn't:

1. `list_resources()` and find the definition you were working on — its
   `v1.cpp` … `vN.cpp` entries and `trajectory.jsonl` are still listed.
2. `read_resource()` its `trajectory.jsonl` and check the last few turns
   for the best-performing version's number and its recorded
   `time_speedup_geomean`/`cycle_speedup_geomean` — the highest `vN.cpp`
   isn't necessarily the best one if a later attempt regressed.
3. `read_resource()` that best `vN.cpp` and `compile()` it again to resume
   from there. It becomes a new `v1` in the fresh session, but it's the
   same code you already had working — you lose the version-history
   numbering, not the optimization progress.
4. For the final §4 report, keep using the *original* starting-point
   numbers from the first `compile`/`evaluate` turns in `trajectory.jsonl`
   — not the numbers from this recovery compile.