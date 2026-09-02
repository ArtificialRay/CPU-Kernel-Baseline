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

Every field's source is either the trajectory (`trajectory.jsonl`, written
by `mcp_app`'s `TrajectoryWriter` — identical format regardless of which
harness drove the session, and server-stamped per row with `ts` + `elapsed_s`)
or a `SessionMetrics` object that the **HarnessAdapter** which ran the job
(`test_scripts/harness_adapters.py`) parsed from its own log format.
`wandb_log_run.py` has no harness-specific parsing of its own — it only knows
the `SessionMetrics`/`TurnRow` shape.

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

### Charts — `turn/*` group (from the trajectory, all harnesses)

Own x-axis (`turn/idx`), one point per agent turn (one server tool call to
the next).

| field | meaning |
|---|---|
| `turn/total_s` | wall time from the previous tool call's row to this one (model thinking + transport + this tool) |
| `turn/llm_s` | `total_s - tool_s` — the model's share |
| `turn/tool_s` | this tool's own execution time (`elapsed_s`, timed in `KernelSession.dispatch_tool_call`) |

The trajectory is the preferred source because it is the same clock and the
same definition of "a turn" for every harness (`turn_timing_source =
"trajectory"` in the summary). For trajectories written before the server
stamped rows, the logger falls back to the harness's own
`SessionMetrics.turn_rows` (`turn_timing_source = "harness-log"`): claude-code
splits its stream-json event timestamps, nanobot uses the per-iteration
timing in its usage sidecar, own uses the timing its loop recorded.

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
| `cost_per_speedup` | `cost_usd / best_speedup` — $ spent per 1x of speedup gained |
| `cost_per_eval` | `cost_usd / n_perf_evals` |
| `best_kernel_version` / `best_kernel_techniques` | winning version + detected optimization idioms (regex match against `TECHNIQUES` in `wandb_log_run.py`: bf16 bfdot/bfmmla, int8 dot/mmla, FMA, prefetch, predication, NEON, unroll, cache blocking) |

### Overview / Runs Table — session fields (from `SessionMetrics`, harness-dependent)

| field | meaning | claude-code | nanobot | own |
|---|---|:---:|:---:|:---:|
| `harness` | which adapter ran the job (also a run tag + config field) | ✅ | ✅ | ✅ |
| `cost_usd` / `cost_source` | total $ spent; `harness` = reported by the CLI, `litellm` = summed per call by our loop, `litellm-estimate` = computed by the logger from the token counts (cache discounts ignored) | ✅ harness | ✅ litellm-estimate | ✅ litellm |
| `num_turns` | agent turns as the harness counts them (claude-code: CLI `num_turns`; nanobot: iterations; own: model calls) | ✅ | ✅ | ✅ |
| `n_tool_calls` | server tool calls in the trajectory — the harness-independent turn count | ✅ | ✅ | ✅ |
| `wall_time_s` | total job wall-clock time | ✅ | ✅ | ✅ |
| `api_retries` | LLM API retries (rate limit/timeout/transient) | ✅ | ✅ | ✅ |
| `session_compile_errors` | compile-error mentions detected in the conversation text (heuristic, distinct from `n_compile_error` which comes from the trajectory) | ✅ | ❌ | ❌ |
| `tokens_input`/`tokens_output`/`tokens_cache_read`/`tokens_cache_created` | token usage (nanobot: cache fields only if its provider reports them) | ✅ | ✅ | ✅ |
| `harness_status` | the harness's own terminal status (nanobot stop reason / own result status) | ❌ | ✅ | ✅ |
| `turn_timing_source` | `trajectory` or `harness-log`, see the `turn/*` section | ✅ | ✅ | ✅ |
| `sec_per_turn_mean`/`median`/`max`, `llm_time_s`, `tool_time_s` | derived from the turn rows | ✅ | ✅ | ✅ |

Where each harness gets its session fields: claude-code parses the CLI's
stream-json log; nanobot parses its `--logs` runtime log plus the
`<log stem>.usage.json` sidecar that `test_scripts/nanobot_run.py` writes
from a nanobot SDK usage hook (the nanobot CLI persists usage nowhere, which
is why the adapter runs that script instead of `nanobot agent`); own reads
the `session` block `eval/evaluator.py::run_agentic_eval` puts in its result
JSON. A ❌ means the harness's log/output doesn't carry that field in a form
we can read; such fields are simply absent from the run, not zero.

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

## Extending session metrics to a new harness

`HarnessAdapter.parse_session_metrics(log_path) -> SessionMetrics` defaults
to an all-empty `SessionMetrics()`. To report real data for a harness,
override it on that harness's adapter class in
`test_scripts/harness_adapters.py` — see the three existing overrides
(each delegates to a module-level `parse_*` function so the manual-backfill
CLI can use it without constructing an adapter). Set `harness` on the
returned object, and leave `cost_usd` at None if the harness doesn't report
cost — the logger estimates it from tokens. `wandb_log_run.py` never needs
to change: it only consumes the `SessionMetrics`/`TurnRow` shape, not any
harness's raw log format. Per-turn latency needs nothing from the harness at
all — it comes from the server-stamped trajectory.
