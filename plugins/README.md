# plugins/

Archived generators that produced what is already in the dataset. Each one is a `kgen discover` plugin: it writes
into a staging directory (`--out-dir`, laid out like `bench-trace/`), and the files go into the warehouse through
`kgen add`, so they pass the same lint as anything an agent writes. New kernels are not added here; the agent
writes the framework-specific pieces at run time.

| Plugin | Produces | Needs |
|---|---|---|
| `simd_loop/plugin.py` | simd-loop definitions, workloads, reference/autovec/baseline-sve/sve2/sme2 solutions (`sme2-kernel:*` tags, M4 streaming entry, n%256 padding bindings) | `dataset/problems/`, `loops/`, clang |
| `llamacpp/plugin.py` | packed-ggml `gemm_ggml_*` definitions, workloads, reference-scalar and baseline-llamacpp-arm solutions | an inventory JSON (run `plugins/llamacpp/inventory.py` as a discover plugin) |

```bash
python -m kgen discover plugins/simd_loop/plugin.py --out report.json -- --out-dir /tmp/stage --loops loop_216
python -m kgen add definition /tmp/stage/definitions/simd-loop/loop_216.json   # likewise workload / solution
```

`llamacpp/test_kquant.py` is the regression test of the k-quant templates (needs a ggml build, see its docstring).
