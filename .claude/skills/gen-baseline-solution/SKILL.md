---
name: gen-baseline-solution
description: Write a baseline solution (the real library's own kernel behind the benchmark ABI) and add it with kgen; use when adding a baseline for a new op type or after definitions change
---

## gen_baseline_solution

A baseline solution calls the real library (ncnn's `*_arm` layers, ggml, KleidiAI) behind the
benchmark's `armbench_entry_<op_type>` ABI. It is the timing baseline for speedups and, because
it is the library's own code, the check that a definition's reference describes the real
operator. Each solution JSON embeds three sources: a contract header, `binding.cpp`
(`armbench_entry_<op_type>`, the ABI shim with per-definition constants baked in) and `kernel.cpp`
(the library call).

You are going to write the solution JSON, using the
closest existing baseline of the same dataset as the pattern, then place it with kgen:

| Command | Purpose |
|---|---|
| `kgen show <def>` | const/var axes, input dtypes, tags, existing solutions |
| `kgen add solution FILE [--replace]` | lint FILE and write `bench-trace/solutions/<dataset>/<author>/<op_type>/<def>.json` |
| `kgen check <def> [--isa ...]` | static lint; on aarch64 builds and runs the baseline against the reference |
| `kgen autovec --isa <tier> --definitions D... [--baseline-author A]` | (aarch64, the tier's own machine) times the baseline against the compiler's auto-vectorized build of the reference-scalar; flags a baseline more than 2x slower |

---

## What to extract from the user's message

- **Op type / definition(s)** - ask if not stated.
- **Backend** - from the definition's `baseline-solution:<backend>` tag. Dataset, author and
  build flags per backend, as they exist today:

| Backend | dataset | author (`config/kernel_contracts.yaml`) | builder | flags |
|---|---|---|---|---|
| ncnn | `ncnn` | `baseline-ncnn-arm` | `NcnnBuilder`, links `libncnn.a` | `-O3 -std=c++17`, link `-fopenmp` |
| llama.cpp | `llama.cpp` | `baseline-llamacpp-arm` | `LlamaCppBuilder`, links ggml static libs | `-O3 -std=c++17` (+ `-march=...` as the dtype needs) |
| KleidiAI | `kleidiai` | `baseline-kleidiai-arm` | see an existing solution | - |

  A new backend needs its builder in `bench/compile/builders/` first; confirm scope with the
  user before starting.
- **Scope** - one definition or every definition carrying the tag.

---

## What a baseline solution JSON looks like

```json
{
  "name":        "baseline-ncnn-arm_<def_name>",
  "definition":  "<def_name>",
  "dataset":     "ncnn",
  "author":      "baseline-ncnn-arm",
  "description": "baseline-ncnn-arm baseline for <def_name>. binding.cpp bakes constexpr params and implements armbench_entry_<op_type> with a void* ABI; kernel.cpp delegates to the backend library. Timing baseline for speedup computation.",
  "spec": {
    "language":        "cpp",
    "target_hardware": ["graviton3", "aarch64-sve", "graviton4", "aarch64-sve2"],
    "entry_point":     "binding.cpp::armbench_entry_<op_type>",
    "dependencies":    [],
    "isa_features":    [],
    "compile_flags":   ["-O3", "-std=c++17"],
    "link_flags":      ["-fopenmp"]
  },
  "sources": [
    { "path": "<op_type>_contract.h", "content": "// contract fn signature" },
    { "path": "binding.cpp",          "content": "// armbench_entry_<op_type>" },
    { "path": "kernel.cpp",           "content": "// contract fn implemented with the library" }
  ]
}
```

- `name` is `<author>_<definition>`; the author must equal the baseline author the benchmark
  is run with (`Benchmark._is_baseline()` compares `solution.author` to it).
- `entry_point` is `binding.cpp::armbench_entry_<op_type>`; pointing at `kernel.cpp` is a
  duplicate-symbol link error.
- The builder takes the baseline path only for `is_baseline` plus the matching dataset; keep
  `dataset` as the table says.
- Bake only **const** axes (and scalar inputs identical across all workloads) as `constexpr`.
  A **var** axis must be derived at run time inside binding.cpp from a tensor's shape, never baked.

Copy the structure from an existing baseline of the same dataset (`bench-trace/solutions/<dataset>/<author>/`).
The argument ABI the dataset adapter builds for each backend (ncnn's void* Mat ABI, ggml's packed
blocks) is in `pitfalls/<backend>.md`, see below.

---

## Pitfalls library (`pitfalls/`, gitignored)

What goes wrong when wrapping one specific library behind the benchmark ABI is kept per backend in
`pitfalls/<backend>.md` (`ncnn.md`, `llamacpp.md`, `kleidiai.md`, ...). The directory is
gitignored, so it is local knowledge that accumulates; it may be missing or empty in a fresh clone.

- **Before writing**, read `pitfalls/<backend>.md` for the backend you are wrapping, if it exists:
  the argument ABI the dataset adapter builds, layer/class choice, fields that are not
  initialized, dtype and scale conventions.
- **While working**, when you hit something that cost you a debugging round (a silent
  wrong-answer, an uninitialized field, a flag the build needs, an ABI detail that differs from
  what the definition suggests), append it to that file before you finish: the symptom, the
  cause, the fix, in a few lines. Create the file for a backend that has none. Keep entries
  specific to the library; anything about kgen or the solution JSON itself belongs in this skill.

---

## Workflow

1. `kgen show <def>`; confirm the `baseline-solution:` tag and read the closest existing baseline.
2. Write `<def>.json` in the scratchpad (a throwaway snippet can render a whole op type from the
   definition; then `add` each file).
3. `python -m kgen add solution <file>`; fix what the sentences name; `--replace` to overwrite.
4. `python -m kgen check <def> --static-only`.
5. On aarch64 (Graviton; `bench-trace/` is gitignored, so rsync it over first):
   `python -m kgen check <def> --num-selected-workloads 5` (add `--isa sve2` where a dataset
   has one baseline author per ISA). A baseline that disagrees with the reference while
   reference-scalar agrees points at the binding or the library call; both disagreeing points at
   the reference.
6. On the tier's own machine: `python -m kgen autovec --isa <tier> --definitions <def>`. A baseline more
   than 2x slower than the auto-vectorized reference-scalar is not an expert baseline: look at the
   binding and the library call (option flags, wrong layer, packing, per-call setup) and fix them, or
   leave the definition out. With several candidate library kernels for one operator, run it with
   `--baseline-author` for each candidate and prefer the one that holds up best against autovec.
   The ratio is a geomean over workloads; the output also names the worst single workload.

---

## Common problems

Library-specific symptoms (uninitialized fields, scale conventions, missing libraries) are in
`pitfalls/<backend>.md`. The general ones:

| Symptom | Cause | Fix |
|---------|-------|-----|
| Garbled args or SIGSEGV on entry | Entry parameter list differs from what the dataset adapter builds for this definition | Re-derive tensor order and dtypes from the adapter's `wrap_inputs` |
| Duplicate symbol at link | `entry_point` points at `kernel.cpp` | `binding.cpp::armbench_entry_<op_type>` |
| `name: expected <author>_<definition>` | Wrong `name` | Fix it |
| `definition ... is not in the warehouse` | Definition not added yet | `kgen add definition` first |
| Baseline is not picked up as the baseline | `author` differs from the baseline author the run uses | Use the author from `config/kernel_contracts.yaml` |
| `check` says baseline matches but scalar does not (or vice versa) | See the diagnosis line | Fix the solution it points at |
