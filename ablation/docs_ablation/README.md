# Docs ablation — does the Arm Software Optimization Guide help?

Three arms of the claude-code fleet, identical except for documentation:

| arm | SWOG MCP resources (`docs/*.md`) on the server | SKILL.md hardware-docs bullet | per-job prompt |
|---|---|---|---|
| `control` | **absent** — deleted from the box's run dir right after the server starts | **stripped** at load time, so the agent isn't told to read files that 404 | unchanged |
| `docs` | present (today's default) | intact | unchanged |
| `nudge` | present | intact | + a strong "read the SWOG first, ground every decision in its latency tables" instruction |

`control` vs `docs` measures whether having the docs matters at all;
`docs` vs `nudge` measures whether the mild mention in SKILL.md is enough
or the agent needs to be pushed. Compare `best_speedup` (and
`iters_to_parity`, `cost_usd`) per arm on W&B.

## Run

```
# one arm = one bench_fleet sweep; same defs, same model/isa, distinct author per arm
python ablation/docs_ablation/run.py --arm control --dataset ncnn --isa sve2 --model sonnet --min-iterations 40
python ablation/docs_ablation/run.py --arm docs    --dataset ncnn --isa sve2 --model sonnet --min-iterations 40
python ablation/docs_ablation/run.py --arm nudge   --dataset ncnn --isa sve2 --model sonnet --min-iterations 40
```

Everything after `--arm` is passed to `test_scripts/bench_fleet.py`
unchanged (`--definitions`, `--max-budget-usd`, `WANDB=1`, ... all work).
`--harness` defaults to `claude-code` and is the only harness supported —
the SKILL.md strip and the prompt nudge are claude-code specific. The author
becomes `<computed author>-<arm>` (e.g. `claude-code-sonnet-sve2-control`)
unless you pass `--author` yourself, and `WANDB_GROUP` defaults to
`docs-ablation-<isa>-<date>` so all arms land in one group.

## How it stays out of the main files

`run.py` touches exactly two seams of `bench_fleet.py`, both by module
attribute swap before calling `bench_fleet.main()`:

- `bench_fleet.ClaudeCodeAdapter` → the arm's subclass from `arms.py`.
  `control` strips the hardware-docs bullet from the SKILL.md text it loads
  (anchored on the bullet's first and last lines — if SKILL.md drifts, this
  fails loudly instead of silently running a docs-exposed "control").
  `nudge` appends `DOC_NUDGE` to each job prompt in `run_job()`.
- `bench_fleet.prepare_session` → (control only) a wrapper that, once the
  remote server is up, runs `rm -rf <run_dir>/docs` on the box and verifies
  it's gone. The server writes `docs/` at startup (`mcp_app/session.py`)
  *before* it starts listening, and `mcp_app/resources.py` globs
  `run_dir/docs/*.md` on every `list_resources()` call, so after the delete
  the docs are neither listed nor readable. No env var, no server flag.

The other arms (`docs`, `nudge`) run the untouched production adapter and
launch path.
