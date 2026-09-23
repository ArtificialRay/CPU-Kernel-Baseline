# ablation/

Documentation only — records of ablation studies run against this repo's
harnesses. No drivers/scripts live here; each study is implemented as a flag
or option on the harness itself (`test_scripts/bench_fleet.py` /
`test_scripts/harness_adapters.py`), not a separate wrapper.

| study | what it varies | how to run |
|---|---|---|
| docs-nudge | whether the agent is forced to read the Arm Software Optimization Guide before optimizing | `test_scripts/bench_fleet.py --harness nanobot --docs-nudge` (nudge) vs. the same command without the flag (control) — see `NanobotAdapter`/`--docs-nudge` in `test_scripts/harness_adapters.py` and `test_scripts/bench_fleet.py` |

Run each arm with its own `--author` so results/solutions don't collide
(`compute_author()` doesn't fold ablation flags into the author string).

Documentation in this repo is classified by ISA: SVE related skills are in `skills/hardware_docs/sve`, SME related skills are in `skills/hardware_docs/sme`