# ablation/

One sub-directory per ablation study. The rule that makes this directory
worth having: **an ablation never edits the main harness files**
(`test_scripts/bench_fleet.py`, `test_scripts/harness_adapters.py`,
`mcp_app/`, `skills/`). It composes them instead — subclassing an adapter,
wrapping a launch step, post-processing a box — so the production sweep path
stays exactly what it is and an ablation can be deleted without leaving
knobs behind.

| study | what it varies | driver |
|---|---|---|
| [`docs_ablation/`](docs_ablation/README.md) | whether the agent has the Arm Software Optimization Guides (and how hard it's pushed to read them) | `python ablation/docs_ablation/run.py --arm {control,docs,nudge} <bench_fleet args>` |

Each driver takes the same arguments as `test_scripts/bench_fleet.py` plus
its own arm selector, derives a distinct `--author` per arm (so results dirs,
boxes and W&B tags never mix), and puts every arm of one study in one W&B
group so the comparison is a single Runs Table filter.
