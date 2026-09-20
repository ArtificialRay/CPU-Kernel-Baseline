#!/usr/bin/env python3
"""Self-contained test for the Q4_K / Q5_K / Q6_K weight-only gemm support.

Run:  python3 scripts/e2e/test_kquant.py           (needs numpy + ml_dtypes +
      the repo's deps; an arm64 Mac/Linux with clang++ and a static ggml build)

Environment:
  LLAMA_CPP_DIR   llama.cpp checkout with ggml/include and a static build in
                  build/ggml/src/libggml{,-base,-cpu}.a (built on demand with
                  `cmake --build build --target ggml ggml-base ggml-cpu`).
  KQUANT_BUILD_DIR  where compiled test .dylib/.so files go (default: a temp dir).

Checks (every one asserts; the script exits non-zero on the first failure):
  A. flat repacks (_repack_q5_k / _repack_q6_k) vs ggml's dequantize_row_qX_K
     on the repacked bytes -- Q5_K against the numpy re-quantized expectation
     (fp16 d/dmin + 6-bit sc/m, same lossy step real Q5_K quantization does)
     and loosely against the raw flat dequant; Q6_K exactly (lossless repack).
  B. packed numpy dequant (kquant_templates.dequant_ggml_blocks, the code the
     layout="ggml" Definition reference embeds) vs ggml, all three types.
  C. reference-scalar kernels (flat q5_k/q6_k, packed q4_k_m/q5_k/q6_k)
     compiled with clang++ -O2 -std=c++14 vs the numpy Definition reference
     at N=64, K=512, M=3 under the harness's gate for these definitions:
     they are tagged `correctness:sqnr`, so bench.runtime.correctness.
     compare_sqnr with min_sqnr_db=20 (BenchmarkConfig default) decides, and
     the elementwise gemm tolerance (abs 2e-3 + rel 1e-2, matched ratio) is
     reported as a diagnostic only -- Q8_K activation quantization makes
     zero-mean dot products miss a 1e-2 relative gate on ~15-20% of elements
     even for the existing bench-trace Q4_K kernel (matched ~0.86 at ~50 dB).
     A stricter 35 dB sanity floor is also asserted: a bit-placement bug
     lands below ~15 dB, Q8_K noise alone sits at 45-52 dB.
  D. baseline-llamacpp-arm kernels (same shapes/layouts) compiled against the
     ggml static libs, driven through LlamaCppDataset.wrap_inputs (so the
     Q5_K/Q6_K triplet detection and the packed uint8 pass-through are
     exercised) vs the numpy reference, same gate.
  E. templating regression: flat q4_k_m definition + both solution JSONs are
     byte-identical to every gemm_q4_k_m_* file already in bench-trace/.
  F. Definition-level round trip for layout="ggml": Definition + Workload
     ({"B": {"type": "bytes", "layout": "ggml_q6_K"}}, M=2) through
     bench.runtime.inputs.gen_inputs_for_workload, exec the reference string,
     output shape [2, N] and finite -- for all three types.
"""

from __future__ import annotations

import ctypes
import glob
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import ml_dtypes  # noqa: E402
import numpy as np  # noqa: E402

from bench.data.definition import Definition  # noqa: E402
from bench.data.workload import Workload  # noqa: E402
from bench.datasets.llama_cpp import (  # noqa: E402
    LlamaCppDataset,
    _kquant_encode_scales_mins,
    _repack_q4_k,
    _repack_q5_k,
    _repack_q6_k,
)
from bench.runtime.correctness import compare_sqnr  # noqa: E402
from bench.runtime.inputs import gen_inputs_for_workload  # noqa: E402
from scripts.e2e import kquant_templates as kt  # noqa: E402

DEFAULT_LLAMA = (
    "/private/tmp/claude-501/-Users-allen-CMU-arm-bench/"
    "4cbdabd8-ae4a-490f-99e4-b317405a9806/scratchpad/llama.cpp"
)
LLAMA = Path(os.environ.get("LLAMA_CPP_DIR", DEFAULT_LLAMA))
GGML_INC = LLAMA / "ggml" / "include"
GGML_LIBS = [LLAMA / "build" / "ggml" / "src" / f"libggml{sfx}.a" for sfx in ("", "-cpu", "-base")]

BUILD = Path(os.environ.get("KQUANT_BUILD_DIR") or tempfile.mkdtemp(prefix="kquant_"))
BUILD.mkdir(parents=True, exist_ok=True)
DYLIB_EXT = ".dylib" if sys.platform == "darwin" else ".so"

# Harness gemm elementwise tolerance (config/kernel_contracts.yaml
# eval_op_type_overrides.gemm) -- diagnostics here; the gate for the
# `correctness:sqnr`-tagged k-quant definitions is SQNR >= MIN_SQNR_DB
# (BenchmarkConfig.min_sqnr_db default).
ABS_TOL, REL_TOL, MATCH_RATIO = 2e-3, 1e-2, 0.98
MIN_SQNR_DB = 20.0
SANITY_SQNR_DB = 35.0

N, K, M = 64, 512, 3
_passed: list = []


def ok(name: str, detail: str = "") -> None:
    _passed.append(name)
    print(f"  PASS {name}" + (f"  ({detail})" if detail else ""))


def ensure_ggml() -> None:
    if not GGML_INC.is_dir():
        sys.exit(f"ggml headers not found at {GGML_INC}; set LLAMA_CPP_DIR")
    if all(p.is_file() for p in GGML_LIBS):
        return
    print(f"  building ggml static libs in {LLAMA / 'build'} ...")
    subprocess.run(
        ["cmake", "--build", "build", "-j8", "--target", "ggml", "ggml-base", "ggml-cpu"],
        cwd=LLAMA, check=True,
    )
    missing = [p for p in GGML_LIBS if not p.is_file()]
    if missing:
        sys.exit(f"ggml libs still missing after build: {missing}")


def compile_shared(out: Path, sources: list, *, flags: list, link: list = ()) -> ctypes.CDLL:
    cmd = ["clang++", "-shared", "-fPIC", *flags, "-o", str(out), *map(str, sources), *map(str, link)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"compile failed: {' '.join(cmd)}\n{r.stderr}")
    return ctypes.CDLL(str(out))


def write_sources(dirname: str, srcs: list) -> Path:
    d = BUILD / dirname
    d.mkdir(parents=True, exist_ok=True)
    for s in srcs:
        (d / s["path"]).write_text(s["content"])
    return d


def harness_match(got: np.ndarray, ref: np.ndarray) -> tuple:
    """bench.runtime.correctness rule: fail iff abs_err > abs_tol AND rel_err > rel_tol."""
    got = np.asarray(got, np.float64)
    ref = np.asarray(ref, np.float64)
    abs_err = np.abs(got - ref)
    rel_err = abs_err / np.maximum(np.abs(ref), 1e-12)
    fail = ~np.isfinite(got) | ((abs_err > ABS_TOL) & (rel_err > REL_TOL))
    ratio = 1.0 - fail.sum() / fail.size
    return ratio, float(abs_err.max()), float(rel_err.max())


def assert_gemm(name: str, got: np.ndarray, ref: np.ndarray) -> None:
    assert got.shape == ref.shape, f"{name}: shape {got.shape} != {ref.shape}"
    c = compare_sqnr(got, ref, min_sqnr_db=MIN_SQNR_DB, abs_tol=ABS_TOL, rel_tol=REL_TOL)
    ratio, max_abs, max_rel = harness_match(got, ref)
    detail = (f"sqnr={c.sqnr_db:.1f} dB; elementwise diag: matched={ratio:.3f} "
              f"max_abs={max_abs:.2e} max_rel={max_rel:.2e}")
    assert c.passed, f"{name}: harness SQNR gate ({MIN_SQNR_DB} dB) failed: {detail}"
    assert c.sqnr_db >= SANITY_SQNR_DB, f"{name}: below {SANITY_SQNR_DB} dB sanity floor: {detail}"
    ok(name, detail)


# ─── random flat inputs (harness-like distributions) ─────────────────────────

def flat_inputs(qt: str, rng: np.random.Generator, n: int, k: int) -> dict:
    if qt == "q4_k_m":
        return {
            "B_q4": rng.integers(0, 256, (n, k // 2), dtype=np.uint8),
            "B_scales": rng.uniform(0.01, 1.0, (n, k // 32)).astype(np.float16),
            "B_mins": rng.uniform(0.01, 1.0, (n, k // 32)).astype(np.float16),
        }
    if qt == "q5_k":
        # bytes 0..255 on purpose: the ABI takes the low 5 bits (the harness's
        # generic uint8 generator emits 1..100).
        return {
            "B_q5": rng.integers(0, 256, (n, k), dtype=np.uint8),
            "B_scales": rng.uniform(0.01, 1.0, (n, k // 32)).astype(np.float16),
            "B_mins": rng.uniform(0.01, 1.0, (n, k // 32)).astype(np.float16),
        }
    return {
        "B_q6": rng.integers(0, 256, (n, k), dtype=np.uint8),  # low 6 bits used
        "B_scales": rng.integers(1, 101, (n, k // 16)).astype(np.int8),
        "B_d": rng.uniform(-1.0, 1.0, (n, k // 256)).astype(np.float16),
    }


def repack(qt: str, fi: dict) -> np.ndarray:
    if qt == "q4_k_m":
        return _repack_q4_k(fi["B_q4"], fi["B_scales"], fi["B_mins"])
    if qt == "q5_k":
        return _repack_q5_k(fi["B_q5"], fi["B_scales"], fi["B_mins"])
    return _repack_q6_k(fi["B_q6"], fi["B_scales"], fi["B_d"])


def run_reference(src: str, **inputs) -> np.ndarray:
    ns: dict = {}
    exec(src, ns)
    return np.asarray(ns["run"](**inputs), dtype=np.float32)


# ─── A/B: dequant checks against ggml ────────────────────────────────────────

def build_shim() -> ctypes.CDLL:
    shim = BUILD / "shim.cpp"
    shim.write_text(
        "#include <cstdint>\n"
        "extern \"C\" {\n"
        "void dequantize_row_q4_K(const void*, float*, int64_t);\n"
        "void dequantize_row_q5_K(const void*, float*, int64_t);\n"
        "void dequantize_row_q6_K(const void*, float*, int64_t);\n"
        "void kq_dq_q4_K(const void* x, float* y, int64_t k) { dequantize_row_q4_K(x, y, k); }\n"
        "void kq_dq_q5_K(const void* x, float* y, int64_t k) { dequantize_row_q5_K(x, y, k); }\n"
        "void kq_dq_q6_K(const void* x, float* y, int64_t k) { dequantize_row_q6_K(x, y, k); }\n"
        "}\n"
    )
    lib = compile_shared(BUILD / f"shim{DYLIB_EXT}", [shim], flags=["-O2"], link=GGML_LIBS)
    for fn in ("kq_dq_q4_K", "kq_dq_q5_K", "kq_dq_q6_K"):
        getattr(lib, fn).argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        getattr(lib, fn).restype = None
    return lib


def ggml_dequant(lib: ctypes.CDLL, qt: str, packed: np.ndarray) -> np.ndarray:
    """packed [rows, nb, bytes] -> float32 [rows, nb*256] via ggml."""
    rows, nb, _ = packed.shape
    k = nb * 256
    packed = np.ascontiguousarray(packed)
    out = np.empty((rows, k), dtype=np.float32)
    fn = getattr(lib, {"q4_k_m": "kq_dq_q4_K", "q5_k": "kq_dq_q5_K", "q6_k": "kq_dq_q6_K"}[qt])
    for r in range(rows):
        fn(packed[r].ctypes.data, out[r].ctypes.data, k)
    return out


def test_repack_vs_ggml(lib: ctypes.CDLL, rng: np.random.Generator) -> None:
    rows, k = 8, 1024
    # Q5_K: lossy 6-bit scale/min encode -> compare to the numpy re-quantized
    # expectation (fp16 d/dmin, 6-bit ls/lm) to fp32 rounding, and loosely to
    # the raw flat dequant.
    fi = flat_inputs("q5_k", rng, rows, k)
    got = ggml_dequant(lib, "q5_k", _repack_q5_k(fi["B_q5"], fi["B_scales"], fi["B_mins"]))
    sc = fi["B_scales"].astype(np.float32).reshape(rows, k // 256, 8)
    mn = fi["B_mins"].astype(np.float32).reshape(rows, k // 256, 8)
    d, dmin, ls, lm, _ = _kquant_encode_scales_mins(sc, mn)
    d16 = d.astype(np.float16).astype(np.float32)
    dmin16 = dmin.astype(np.float16).astype(np.float32)
    q = (fi["B_q5"] & 0x1F).astype(np.float32)
    ds = np.repeat((d16[..., None] * ls).reshape(rows, -1), 32, axis=-1)
    dm = np.repeat((dmin16[..., None] * lm).reshape(rows, -1), 32, axis=-1)
    expect = ds * q - dm
    err = np.abs(got - expect).max()
    assert err <= 1e-5 * np.abs(expect).max(), f"q5_k repack vs requantized expectation: max err {err}"
    flat = run_reference(kt.def_reference("q5_k"), A=np.eye(k, dtype=ml_dtypes.bfloat16),
                         B_q5=fi["B_q5"], B_scales=fi["B_scales"], B_mins=fi["B_mins"]).T
    ferr = np.abs(got - flat).max()
    assert ferr <= 0.02 * np.abs(flat).max(), f"q5_k repack vs flat dequant: max err {ferr}"
    ok("A. _repack_q5_k vs ggml dequantize_row_q5_K", f"max err vs requantized {err:.2e}, vs flat {ferr:.2e}")

    # Q6_K: lossless repack -> exact to fp32 rounding.
    fi = flat_inputs("q6_k", rng, rows, k)
    got = ggml_dequant(lib, "q6_k", _repack_q6_k(fi["B_q6"], fi["B_scales"], fi["B_d"]))
    flat = run_reference(kt.def_reference("q6_k"), A=np.eye(k, dtype=ml_dtypes.bfloat16),
                         B_q6=fi["B_q6"], B_scales=fi["B_scales"], B_d=fi["B_d"]).T
    err = np.abs(got - flat).max()
    assert err <= 1e-5 * np.abs(flat).max(), f"q6_k repack vs flat dequant: max err {err}"
    ok("A. _repack_q6_k vs ggml dequantize_row_q6_K", f"max err {err:.2e} (lossless)")


def test_packed_dequant_vs_ggml(lib: ctypes.CDLL, rng: np.random.Generator) -> None:
    rows, k = 8, 1024
    for qt in kt.QTYPES:
        packed = repack(qt, flat_inputs(qt, rng, rows, k))
        got = ggml_dequant(lib, qt, packed)
        mine = kt.dequant_ggml_blocks(qt, packed.reshape(rows, -1))
        err = np.abs(got - mine).max()
        assert mine.shape == (rows, k)
        assert err <= 1e-5 * np.abs(got).max(), f"{qt} dequant_ggml_blocks vs ggml: max err {err}"
        ok(f"B. dequant_ggml_blocks({qt}) vs ggml", f"max err {err:.2e}")


# ─── C/D: compiled kernels vs numpy reference ────────────────────────────────

def make_case(qt: str, layout: str, rng: np.random.Generator) -> tuple:
    """(Definition, np_inputs dict, reference output [M, N])."""
    A = rng.uniform(-1.0, 1.0, (M, K)).astype(ml_dtypes.bfloat16)
    fi = flat_inputs(qt, rng, N, K)
    if layout == "flat":
        inputs = {"A": A, **fi}
    else:
        inputs = {"A": A, "B": repack(qt, fi).reshape(N, -1)}
    d = Definition.model_validate(
        kt.definition_json(qt, N, K, description="test", tags=[], layout=layout)
    )
    ref = run_reference(d.reference, **inputs)
    return d, inputs, ref


def test_reference_scalar(qt: str, layout: str, rng: np.random.Generator) -> None:
    d, inputs, ref = make_case(qt, layout, rng)
    srcdir = write_sources(f"rs_{qt}_{layout}", kt.reference_scalar_sources(qt, N, K, layout))
    spec = kt.spec(qt, "reference-scalar")
    lib = compile_shared(srcdir / f"lib{DYLIB_EXT}", [srcdir / "gemm.cpp", srcdir / "kernel.cpp"],
                         flags=spec["compile_flags"])
    out = np.zeros((M, N), dtype=np.float32)
    names = [n for n in d.inputs if n != "A"]
    bufs = [np.ascontiguousarray(inputs[n]) for n in names]
    A = np.ascontiguousarray(inputs["A"])
    fn = lib.armbench_entry_gemm
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_void_p] * len(bufs) + [ctypes.c_int]
    rc = fn(A.ctypes.data, out.ctypes.data, *[b.ctypes.data for b in bufs], M)
    assert rc == 0
    assert_gemm(f"C. reference-scalar {qt}/{layout} ({d.name})", out, ref)


def test_baseline(qt: str, layout: str, rng: np.random.Generator) -> None:
    d, inputs, ref = make_case(qt, layout, rng)
    srcdir = write_sources(f"bl_{qt}_{layout}", kt.baseline_sources(qt, N, K, layout))
    spec = kt.spec(qt, "baseline-llamacpp-arm")
    lib = compile_shared(
        srcdir / f"lib{DYLIB_EXT}", [srcdir / "binding.cpp", srcdir / "kernel.cpp"],
        flags=[*spec["compile_flags"], f"-I{GGML_INC}"], link=[*GGML_LIBS, "-lpthread"],
    )
    fn = lib.armbench_entry_gemm
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    ds = LlamaCppDataset()
    ctx = ds.wrap_inputs(inputs, "gemm", lib, definition=d, out_shape=ref.shape)
    # the adapter's slot layout: flat -> [A, packed, NULL, NULL]; ggml -> [A, B]
    slots = list(ctx.entry_args[0])
    if layout == "flat":
        assert len(slots) == 4 and slots[2] is None and slots[3] is None, slots
    else:
        assert len(slots) == 2 and slots[1] is not None, slots
    rc = fn(*ctx.entry_args)
    assert rc == 0
    out = ds.unwrap_output(ctx).copy()
    ds.release(ctx)
    assert_gemm(f"D. baseline-llamacpp-arm {qt}/{layout} ({d.name})", out, ref)


# ─── E: templating regression ────────────────────────────────────────────────

def test_q4_regression() -> None:
    defs = sorted(glob.glob(str(REPO / "bench-trace/definitions/gemm/gemm_q4_k_m_*.json")))
    assert defs, "no gemm_q4_k_m definitions found in bench-trace/"
    n_files = 0
    for p in defs:
        d = json.load(open(p))
        n_, k_ = d["axes"]["N"]["value"], d["axes"]["K"]["value"]
        gen = kt.definition_json("q4_k_m", n_, k_, description=d["description"], tags=d["tags"])
        assert gen == d, f"definition_json differs from {p}"
        n_files += 1
        for author in kt.AUTHORS:
            sp = REPO / f"bench-trace/solutions/llama.cpp/{author}/gemm/gemm_q4_k_m_n{n_}_k{k_}.json"
            s = json.load(open(sp))
            gen = kt.solution_json("q4_k_m", author, n_, k_)
            assert gen == s, f"solution_json({author}) differs from {sp}"
            n_files += 1
    ok("E. q4_k_m templating byte-identical to bench-trace", f"{n_files} files")


# ─── F: Definition-level round trip through the harness input generator ──────

def test_definition_roundtrip() -> None:
    for qt in kt.QTYPES:
        tag = kt.quant_tag(qt, "ggml")
        d = Definition.model_validate(
            kt.definition_json(qt, N, K, description="test", tags=[], layout="ggml")
        )
        w = Workload.model_validate({
            "axes": {"M": 2},
            "inputs": {"A": {"type": "random"}, "B": {"type": "bytes", "layout": tag}},
        })
        inputs = gen_inputs_for_workload(d, w)
        k_bytes = K // 256 * {"q4_k_m": 144, "q5_k": 176, "q6_k": 210}[qt]
        assert inputs["B"].dtype == np.uint8 and inputs["B"].shape == (N, k_bytes), inputs["B"].shape
        assert inputs["A"].shape == (2, K)
        out = run_reference(d.reference, **inputs)
        assert out.shape == (2, N), out.shape
        assert np.all(np.isfinite(out))
        w_dq = kt.dequant_ggml_blocks(qt, inputs["B"])
        ok(f"F. Definition round trip {d.name} bytes/{tag}",
           f"C shape {out.shape}, |C| max {np.abs(out).max():.3f}, |W| max {np.abs(w_dq).max():.3f}")


def main() -> int:
    print(f"build dir: {BUILD}")
    ensure_ggml()
    rng = np.random.default_rng(1234)
    shim = build_shim()
    test_repack_vs_ggml(shim, rng)
    test_packed_dequant_vs_ggml(shim, rng)
    for layout, qts in (("flat", ("q5_k", "q6_k")), ("ggml", kt.QTYPES)):
        for qt in qts:
            test_reference_scalar(qt, layout, rng)
    for layout, qts in (("flat", ("q5_k", "q6_k")), ("ggml", kt.QTYPES)):
        for qt in qts:
            test_baseline(qt, layout, rng)
    test_q4_regression()
    test_definition_roundtrip()
    print(f"\nALL {len(_passed)} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
