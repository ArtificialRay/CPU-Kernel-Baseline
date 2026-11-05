# End-to-end: agent kernels inside llama.cpp

The rest of the benchmark measures one kernel at a time. `e2e` answers the follow-up question: take the
kernels an agent wrote for a real model's matmuls, splice them into llama.cpp, and measure tokens/s and
perplexity.

## Before you run it

- **Definitions** tagged `e2e:<model>` in `bench-trace/` (the model is a key of `config/e2e_models.json`).
  The scripts in `scripts/e2e/` make them for a new model.
- **Agent solutions** under `bench-trace/solutions/llama.cpp/<author>/gemm/`: run the agents with
  `test_scripts/bench_fleet.py --dataset llama.cpp --definitions "<names>" --sync-solutions ...`.
- **A Graviton box** registered in `provisioning/eval_config.json` under the label `e2e-<model>-<isa>`
  (`--label` picks another), or let `deploy` create it with `--provision`.

## Usage

```
python -m e2e evaluate --model qwen3.5-4b --author AUTHOR --rows-abi
```

`evaluate` runs four steps in order. Add `--steps a,b` to run only some of them; a step whose result is
already in `e2e-runs/llamacpp-<model>-<isa>/` is skipped (`--force` redoes it).

| Step | What it does | Main options |
|---|---|---|
| `kernel_set` | Compiles the author's submitted kernels into shared objects and writes the manifest that tells llama.cpp which matmuls to replace. | `--author` (required), `--kernel all\|NAME`, `--rows-abi`, `--reference-author` with `--expect-changed`, `--expect-same`, `--expect-count`, `--forbid` |
| `deploy` | On the box: builds the pinned llama.cpp tools tree, downloads the model, and with `--agent-build` builds the variant with the replacement hook. | `--label`, `--provision`, `--agent-build` |
| `measure` | **Runs on the box** (printed, not run, from elsewhere): llama-bench for each build, interleaved and repeated, and llama-perplexity. | `--perplexity FILE`, `--threads`, `--pp`, `--tg`, `--reps`, `--fa` |
| `report` | Prints throughput, speedup and perplexity of the builds as one table. If the results file is missing it is rebuilt from `--log`, the output `measure` printed. | `--baseline`, `--log` |

- `--kernel all` (default) uses every submitted definition tagged `e2e:<model>`; `--kernel NAME` just that one.
- `--rows-abi` lets the hook split a matmul's rows across ggml's threads.
- The guards compare the set with a previous author's: `--reference-author PREV --expect-changed DEF...`
  fails if a kernel you meant to change is byte-identical to PREV's.
- Control run: `--author reference-scalar --rows-abi` builds the scalar set, which should reproduce stock perplexity.

On the box:

```
python -m e2e evaluate --model qwen3.5-4b --steps measure --perplexity wiki.test.raw
```

`deploy` only creates a box with `--provision`; tearing one down stays with
`provisioning/provision.py --teardown --label ...`.

## What is compared

Three builds: 
- **stock**: llamacpp default setting on this e2e model 
- **norepack**: llamacpp with ggml's repack and KleidiAI paths off
- **agent**: the norepack configuration plus the override hook, which replaces only the matmuls listed in the manifest with the agent's kernels

Details of the hook: `e2e/override/README.md`.
