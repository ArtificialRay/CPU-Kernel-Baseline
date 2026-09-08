# analysis/

Operational tooling for driving and instrumenting large batch sweeps on top
of `test_scripts/bench_fleet.py`. This README documents what
`wandb_log_run.py` actually writes to Weights & Biases and where to find it
on the run page; the other two files are one-off sweep-orchestration scripts
(see their own docstrings).

- `wandb_log_run.py` — per-definition W&B logging, imported directly by
  `bench_fleet.py` (`log_run_to_wandb()`). Not a CLI/subprocess in the
  production path; its `__main__` is a thin manual-backfill entry point only.
- `resume_sweep.sh` — idempotent multi-round driver for one specific sweep
  (cost-sorted, cross-dataset, stall-detecting). Hardcodes that sweep's own
  parameters (author, paths); not a generic tool.
- `daemonize_sweep.py` — double-fork+setsid launcher so `resume_sweep.sh`
  survives the launching shell closing.

## What ends up on the W&B run page, and where

One W&B run = one definition. Fields fall into two shapes: **time series**
(logged with `wandb.log(..., step=...)`, shown in the **Charts** panel) and
**scalars** (`run.summary`/`config`, shown in the **Overview** tab or as
columns you add to the **Runs Table** — they do *not* appear as charts).

Every field's source is the trajectory (`trajectory.jsonl`, written by
`mcp_app`'s `TrajectoryWriter` — identical format regardless of which
harness drove the session).

### Charts — per-evaluate curve (from trajectory, all harnesses)

One point per perf-eval call, x-axis = `iteration`.

| field | meaning |
|---|---|
| `time_speedup` | this eval's time speedup vs baseline (geomean) |
| `cycle_speedup` | same, in CPU cycles |
| `best_so_far` | running max of `time_speedup` up to this point |
| `marginal_gain` | `best_so_far` minus the previous point's `best_so_far` |
| `ipc` | instructions-per-cycle (hardware perf counter mean) |
| `cache_misses` | cache-miss counter mean |
| `max_abs_error` / `max_rel_error` | this version's worst correctness-check error |

### Overview / Runs Table — run summary (from trajectory, all harnesses)

Scalars, not charts. Taxonomy counts every `evaluate`/`compile` call in the
trajectory (not just perf-mode ones), so it's accurate regardless of harness.

| field | meaning |
|---|---|
| `best_speedup` | max `time_speedup` over the whole run |
| `best_version_iteration` | which perf-eval point achieved it |
| `n_perf_evals` | perf-mode eval count (curve length) |
| `n_evaluations` | total eval calls, all modes |
| `final_status` | status of the last eval call |
| `starting_speedup` | first perf-eval's speedup — usually v1/naive-scalar vs baseline |
| `baseline_vs_scalar` | `1 / starting_speedup` — how much faster the baseline is than naive scalar code |
| `weak_baseline` | `baseline_vs_scalar < 2.0` — baseline barely beats naive code; treat this def's speedups with caution |
| `iters_to_parity` | first perf-eval index where `best_so_far >= 1.0` |
| `iters_to_plateau` | first index where `best_so_far >= 0.98 * best_speedup` |
| `n_passed` / `n_incorrect` / `n_runtime_error` / `n_timeout` | eval-call taxonomy (status = `PASSED` / `INCORRECT_NUMERICAL` / `RUNTIME_ERROR` / `TIMEOUT`) |
| `n_compile_error` | compile-call taxonomy |
| `worst_max_abs_error` / `worst_max_rel_error` | worst correctness error across all versions |
| `best_kernel_version` / `best_kernel_techniques` | winning version + detected optimization idioms (regex match against `TECHNIQUES` in `wandb_log_run.py`: bf16 bfdot/bfmmla, int8 dot/mmla, FMA, prefetch, predication, NEON, unroll, cache blocking) |

### Config (from `bench_fleet.py`, all harnesses)

| field | meaning |
|---|---|
| `definition` / `dataset` / `op_type` / `isa` / `model` / `author` | run identity |
| `instance_type` | `WANDB_INSTANCE_TYPE` env var, if set |
| `baseline_kernel_sha` | sha256(first 12 hex chars) of the baseline `kernel.cpp` this run was scored against — lets you tell runs apart across baseline changes |

### Table / Artifact (from trajectory + kernel sources, all harnesses)

- `kernels` table: one row per version (`v1`, `v2`, ...) with its speedup,
  detected techniques, and full source — browsable in the run page.
- A versioned `kernel` W&B Artifact bundling every `vN.cpp` (+ `vN.s` if
  disassembled) and the full `trajectory.jsonl`.
