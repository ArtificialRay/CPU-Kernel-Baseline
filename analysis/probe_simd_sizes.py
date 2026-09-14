#!/usr/bin/env python3
"""Call a simd-loop baseline-sve kernel directly at many axis sizes, in a
forked child per size, with 0xA5 canary padding after every buffer: reports
per size whether the child crashed (signal), overran a buffer (canary
clobbered), mismatched the Python reference, or passed. Run on a box:
  python analysis/probe_simd_sizes.py loop_113 [sizes...]"""
import ctypes, json, os, re, sys, signal
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent; BT = ROOT / "bench-trace"
PAD = 4096
NP = {"int8": np.int8, "uint8": np.uint8, "int16": np.int16, "uint16": np.uint16, "int32": np.int32,
      "uint32": np.uint32, "int64": np.int64, "uint64": np.uint64, "float16": np.float16,
      "float32": np.float32, "float64": np.float64}

def padded(shape, dtype, rng=None):
    n = int(np.prod(shape)) if len(shape) else 1
    raw = np.full(n * np.dtype(dtype).itemsize + PAD, 0xA5, np.uint8)
    view = raw[:n * np.dtype(dtype).itemsize].view(dtype).reshape(shape)
    if rng is not None:
        # Same distributions as bench/runtime/inputs.py::_gen_random_tensor
        # (the evaluator's contract): floats uniform(-1, 1), integers in [1, 100].
        if np.issubdtype(dtype, np.floating):
            view[...] = rng.uniform(-1, 1, shape).astype(dtype)
        else:
            view[...] = rng.integers(1, 101, shape).astype(dtype)
    else:
        view[...] = 0
    return raw, view

def main():
    # optional: --author <name> (default baseline-sve), --so <prebuilt .so> (skip the build,
    # e.g. when running under an emulator), then --workloads | --axes ... | sizes...
    argv = sys.argv[1:]; author = "baseline-sve"; so_arg = None
    while len(argv) > 2 and argv[1] in ("--author", "--so"):
        if argv[1] == "--author": author = argv[2]
        else: so_arg = argv[2]
        argv = [argv[0]] + argv[3:]
    sys.argv = [sys.argv[0]] + argv
    lid = sys.argv[1]; wl_mode = len(sys.argv) > 2 and sys.argv[2] == "--workloads"
    sizes = [int(s) for s in sys.argv[2:]] if not wl_mode and not (len(sys.argv) > 2 and sys.argv[2] == "--axes") else []
    sizes = sizes or [1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 256, 1000, 1024]
    d = json.loads((BT / "definitions/simd-loop" / f"{lid}.json").read_text())
    sp = BT / "solutions/simd-loop" / author / lid / f"{author}_{lid}.json"
    solj = json.loads(sp.read_text())
    if so_arg:
        so = Path(so_arg)
    else:
        from bench.data.solution import Solution
        from bench.data.definition import Definition
        from bench.compile.builders.simd_loop import SimdLoopBuilder
        sol = Solution.model_validate(solj)
        so = SimdLoopBuilder().build(Definition.model_validate(d), sol).so_path
    hdr = next(s["content"] for s in solj["sources"] if s["path"] == f"{lid}.h")
    sig = re.search(rf"armbench_entry_{lid}\s*\(([^)]*)\)", hdr).group(1)
    params = [p.strip().split()[-1].lstrip("*") for p in sig.split(",")]
    ptypes = [ctypes.c_void_p if "void" in p else ctypes.c_int64 for p in sig.split(",")]
    ns = {}; exec(d["reference"], ns); ref = ns["run"]
    var_axes = [a for a, spec in d["axes"].items() if spec["type"] == "var"]
    const_axes = {a: spec["value"] for a, spec in d["axes"].items() if spec["type"] == "const"}
    print(f"{lid} entry({sig.strip()}) var_axes={var_axes}", flush=True)
    cases = [{a: s for a in var_axes} for s in sizes]
    if wl_mode:
        cases = [json.loads(l)["axes"] for l in (BT / "workloads/simd-loop" / f"{lid}.jsonl").read_text().splitlines() if l.strip()]
    if len(sys.argv) > 2 and sys.argv[2] == "--axes":
        cases = [{kv.split("=")[0]: int(kv.split("=")[1]) for kv in a.split(",")} for a in sys.argv[3:]]
    for case in cases:
        s = case
        r, w = os.pipe(); pid = os.fork()
        if pid == 0:
            os.close(r)
            try:
                axes = {**const_axes, **case}
                rng = np.random.default_rng(1234 + sum(case.values()))
                bufs, args = {}, {}
                for name, spec in d["inputs"].items():
                    shape = [axes[x] if isinstance(x, str) else x for x in spec["shape"]]
                    raw, view = padded(shape, NP[spec["dtype"]], rng); bufs[name] = (raw, view); args[name] = view
                (oname, ospec), = d["outputs"].items()
                oshape = [] if ospec["shape"] is None else [axes[x] if isinstance(x, str) else x for x in ospec["shape"]]
                oraw, oview = padded(oshape, NP[ospec["dtype"]]); bufs[oname] = (oraw, oview)
                lib = ctypes.CDLL(str(so)); fn = getattr(lib, f"armbench_entry_{lid}"); fn.argtypes = ptypes; fn.restype = ctypes.c_int
                call = []; scratch = {}
                inplace = "res_out" not in params          # in-place sorts: output is the data buffer
                for p, t in zip(params, ptypes):
                    if p == "res_out": call.append(oview.ctypes.data)
                    elif p in args: call.append(args[p].ctypes.data)
                    elif t is ctypes.c_int64 and p != "unused":
                        call.append(int(next(v for k, v in axes.items() if k.lower() == p.lower())))
                    elif p == "unused": call.append(0)
                    else:  # scratch pointer (temp/hist/prfx/block_sizes): generous zeroed buffer + canary
                        raw, view = padded([max(int(np.prod(oview.shape)) if oview.shape else 1, 4096) * 32], np.uint8)
                        scratch[p] = (raw, view); bufs["scratch_" + p] = (raw, view); call.append(view.ctypes.data)
                expected = ref(*[args[n] for n in d["inputs"]])
                fn(*call)
                overrun = [n for n, (raw, view) in bufs.items() if not np.all(raw[view.nbytes:] == 0xA5)]
                if inplace: oview = args[next(iter(d["inputs"]))]
                exp = np.asarray(expected).reshape(oview.shape).astype(oview.dtype)
                if np.issubdtype(oview.dtype, np.floating):
                    ok = np.allclose(oview.astype(np.float64), exp.astype(np.float64), rtol=1e-3, atol=1e-3, equal_nan=True)  # evaluator tolerances
                else:
                    ok = np.array_equal(oview, exp)
                bad = int(np.sum(~np.isclose(oview.astype(np.float64), exp.astype(np.float64), rtol=1e-3, atol=1e-3, equal_nan=True))) if not ok else 0
                msg = ("PASS" if ok else f"MISMATCH({bad}/{oview.size} elems)") + (f" OVERRUN{overrun}" if overrun else "")
                if not ok:
                    idx = np.argwhere(oview.reshape(-1) != exp.reshape(-1)).reshape(-1)[:4]
                    msg += " e.g. " + ", ".join(f"[{i}] got={oview.reshape(-1)[i]} exp={exp.reshape(-1)[i]}" for i in idx)
            except Exception as e:
                msg = f"EXC {type(e).__name__}: {str(e)[:80]}"
            os.write(w, msg.encode()); os._exit(0)
        os.close(w); out = os.read(r, 4096).decode(); os.close(r)
        _, status = os.waitpid(pid, 0)
        if os.WIFSIGNALED(status): out = f"CRASH {signal.Signals(os.WTERMSIG(status)).name}"
        print(f"  {json.dumps(case)}: {out}", flush=True)

if __name__ == "__main__":
    main()
