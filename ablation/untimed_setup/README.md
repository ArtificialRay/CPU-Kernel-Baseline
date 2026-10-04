# untimed_setup (branch `ablation/untimed-setup`)

**What it varies:** whether a kernel's one-time preparation (weight
repacking, layout transforms, building a library layer and its pipeline) is
charged to every timed call. On `main` it is; on this branch it is not, for
the expert baselines and the agents' candidates alike.

Unlike the studies in `ablation/` on `main`, this one changes the harness
itself, so it lives on its own branch.

## How it works

- A kernel may export `armbench_setup_<op_type>` (the entry's parameters) and
  `armbench_teardown_<op_type>(void)`. The evaluator calls setup once per
  workload, before the correctness call and the timed loop, and teardown after
  the last call (`bench/evaluators/evaluator.py`). Neither is timed.
- Setup must not be able to precompute a result (issue #85). It gets the
  entry's argument list in which only the inputs declared as weights in
  `config/kernel_contracts.yaml::setup_weights` are real; every other input
  and the output are zero-filled stand-ins with the same shapes, and scalar
  and shape arguments are unchanged. Definitions without declared weights
  (mha, gqa, mla_prefill, pooling, SIMD loops) never get setup called;
  for mla_decode only `v_mla` counts as a weight.
- Agents see a `<definition>/setup-hook.md` resource next to
  `reference-scalar-kernel.cpp` with the exact signature and the weight list;
  the harness SKILL.md files and the own-harness prompt point to it.
- `EvalConfig.timing_protocol` ends in ` setup=untimed` on this branch, so a
  baseline collected on `main` is never reused here (and the other way round):
  every instance re-collects its baselines on first use.

## Baselines

The baselines live in the dataset, not in this repo. Before collecting
baselines on an instance, add the hook to them in that instance's
`bench-trace/`:

```bash
python3 ablation/untimed_setup/add_setup_hooks.py --root bench-trace --check   # what would change
python3 ablation/untimed_setup/add_setup_hooks.py --root bench-trace           # rewrite in place
```

| baselines | one-time work timed on `main` | after the script |
|---|---|---|
| `baseline-ncnn-arm`, conv2d and conv2d_depthwise (18) | build `Convolution(DepthWise)_arm` and run `create_pipeline` (weight transform) on every call | setup builds the layer once and keeps it; each call runs `forward` and copies the output |
| `baseline-kleidiai-arm` (10) | `memcmp` of the weights against the cached packed copy on every call | setup packs once; the check is skipped until teardown |
| `baseline-llamacpp-arm` | none: each one already caches its ggml graph after the first call, which is the untimed correctness call | unchanged |
| ncnn pooling, SIMD loops | no weights | unchanged; setup is never called |

Both hooks call the baseline's own entry once from setup with the arguments
setup receives, with a flag set that makes the entry keep what it prepared.
The script is idempotent.

Note for reading results: the llama.cpp baselines key their caches on input
pointers, and the harness passes the same pointers to every timed call, so
they also skip per-call activation conversions (e.g. widening A from bf16).
That holds on `main` too and makes those baselines faster than a cold call.

## Testing to do

Nothing on this branch has been run yet.

1. **Build and correctness.** On a c7g.xlarge, run the script on
   `bench-trace/`, then `python3 -m bench.cli collect-baselines
   --baseline-author baseline-ncnn-arm` and `--baseline-author
   baseline-kleidiai-arm`: every ncnn conv2d / conv2d_depthwise workload and
   the 3 SVE KleidiAI kernels must pass. The 7 SME2 KleidiAI kernels need
   mac-m4.metal. The ncnn rewrite has not been compiled yet; the KleidiAI
   bindings pass a clang syntax check.
2. **Effect on the baselines.** Compare each workload's baseline `min_ns`
   with the same baseline collected on `main` on the same instance type.
3. **Agent path.** A short session on one gemm and one conv2d definition:
   `setup-hook.md` is listed and readable, a kernel that exports setup runs
   and is correct, and a setup that reads an activation sees only zeros.
4. **Results.** Re-time the stored final kernel of each run on this branch
   (same extractor as for re-timing under the new timing protocol), or rerun
   agents with a distinct `--author` / W&B group suffix such as
   `__untimed_setup`, and rebuild the tables.
