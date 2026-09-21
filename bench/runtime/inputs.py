"""Input generators for benchmark workloads.

`gen_inputs_for_workload` is the main entry point — it reads each input's type
from `workload.inputs` and either returns the scalar value directly or generates
a bounded-uniform random tensor seeded from the workload's uuid.

The legacy `make_weights` / `make_mat_ramp` / `make_mat_ramp_2d` functions are
kept for standalone testing but are no longer called by the harness.
"""

import hashlib
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import ml_dtypes
import numpy as np

from bench.data.definition import AxisConst, Definition, DType
from bench.data.workload import Workload


# ── Legacy bit-exact ports of ncnn_helpers.h (kept for standalone use) ────────

def make_weights(n: int, scale: float = 1.0) -> np.ndarray:
    """Port of ncnn_helpers.h `make_weights` (LCG, values in [-0.5, 0.5]*scale)."""
    i = np.arange(n, dtype=np.int64)
    lcg = (i * 1234567 + 7654321) % 1000
    return (lcg.astype(np.float32) / 1000.0 - 0.5).astype(np.float32) * np.float32(scale)


def make_mat_ramp(shape_chw: Tuple[int, int, int]) -> np.ndarray:
    """Port of ncnn_helpers.h `make_mat_ramp` (values 0.1..10.0 cycling, CHW layout)."""
    c, h, w = shape_chw
    idx = np.arange(c * h * w, dtype=np.int64)
    return (((idx % 100) + 1).astype(np.float32) * np.float32(0.1)).reshape(c, h, w)


def make_mat_ramp_2d(shape_hw: Tuple[int, int]) -> np.ndarray:
    """Port of ncnn_helpers.h `make_mat_ramp_2d` (values 0.1..10.0 cycling, HW layout)."""
    h, w = shape_hw
    idx = np.arange(h * w, dtype=np.int64)
    return (((idx % 100) + 1).astype(np.float32) * np.float32(0.1)).reshape(h, w)


# ── Random input generation ────────────────────────────────────────────────────

def _uuid_to_seed(uuid_str: str) -> int:
    """Convert a workload uuid string to a 32-bit rng seed."""
    try:
        return int(uuid_str, 16) % (2**32)
    except ValueError:
        return int(hashlib.md5(uuid_str.encode()).hexdigest(), 16) % (2**32)


def _gen_random_tensor(
    shape: tuple, dtype, rng: np.random.Generator, *, non_negative: bool = False
) -> np.ndarray:
    """Generate a bounded random tensor suitable for kernel benchmarking.

    Float tensors use uniform(-1, 1) to avoid catastrophic cancellation in
    long reductions; integer tensors use integers in [1, 100]. `non_negative`
    generates uniform(0.01, 1) instead — for block-quantization scale/min
    tensors (e.g. Q4_K's `*_scales`/`*_mins`), which are always non-negative
    magnitudes in real quantization. This matters beyond realism: some
    encode paths (e.g. LlamaCppDataset._repack_q4_k, which re-derives 6-bit
    scale/min factors the way real GGML does) clamp negative inputs to 0,
    silently discarding data if this constraint isn't honored.
    """
    if dtype == ml_dtypes.bfloat16 or np.issubdtype(dtype, np.floating):
        lo = 0.01 if non_negative else -1.0
        return rng.uniform(lo, 1.0, shape).astype(dtype)
    elif np.issubdtype(dtype, np.integer):
        return rng.integers(1, 101, shape).astype(dtype)
    else:  # bool
        return rng.integers(0, 2, shape, dtype=np.uint8).astype(np.bool_)


_GGML_KQUANT_BLOCK_BYTES = {"ggml_q4_K": 144, "ggml_q5_K": 176, "ggml_q6_K": 210, "ggml_q8_0": 34}
_GGML_KQUANT_BLOCK_ELEMS = {"ggml_q4_K": 256, "ggml_q5_K": 256, "ggml_q6_K": 256, "ggml_q8_0": 32}


def _gen_ggml_kquant_rows(shape: tuple, layout: str, rng: np.random.Generator) -> np.ndarray:
    """Random k-quant weight rows in ggml's packed block layout, produced the way
    ggml produces them: draw Gaussian float weights, then actually quantize them.

    `shape` is [N, K_bytes] with K_bytes = K/256 * sizeof(block_qX_K) (Q8_0:
    K/32 * 34).

    This used to draw the quant integers uniform over the type's range and the
    per-sub-block scale and min *independently from the same distribution*. That
    produces block layouts no quantizer would ever emit. In a real Q4_K tensor the
    sub-block min is about 2.7 sigma of the weight distribution while one quant step
    is about 0.36 sigma, so `dmin` runs ~8x `d` and the 6-bit mins sit near 46/63 --
    the min-correction term carries far more of the weight value than the scaled
    quants do. Under the old uniform draw `dmin/d` was 1.0 and mins averaged 37, which
    made the min term a minor correction and hid any kernel that mishandled it. The
    2026-09-21 Fable kernels did mishandle it (int8 block sums under a batch-shared
    exponent) and the harness could not see it; see docs/e2e_qwen35.md.

    Quantizing real weights reproduces the true relationship for free: per 32-element
    sub-block take min/max, step = (max-min)/levels, and let the repack helpers do
    ggml's own 6-bit encode of the per-sub-block scales and mins. Q6_K and Q8_0 have
    no min term and are quantized symmetrically about zero the way ggml does.
    sigma is chosen so dequantized weights land at O(0.01-0.1), matching a real GGUF.
    """
    from bench.datasets.llama_cpp import _repack_q4_k, _repack_q5_k, _repack_q6_k, _repack_q8_0

    if len(shape) != 2:
        raise ValueError(f"bytes layout {layout!r} needs a 2-D [N, K_bytes] shape, got {shape}")
    n_rows, k_bytes = int(shape[0]), int(shape[1])
    blk = _GGML_KQUANT_BLOCK_BYTES[layout]
    if k_bytes % blk != 0:
        raise ValueError(
            f"bytes layout {layout!r}: last dim {k_bytes} is not a multiple of the "
            f"{blk}-byte block size"
        )
    nb = k_bytes // blk
    k = nb * _GGML_KQUANT_BLOCK_ELEMS[layout]
    SIGMA = 0.009  # weight std dev, matched to a real Qwen3.5-4B Q4_K_M tensor
    w = rng.normal(0.0, SIGMA, (n_rows, k)).astype(np.float32)

    def _asym(levels: int, group: int):
        """ggml's asymmetric k-quant of `group`-element sub-blocks onto [0, levels].

        Returns (q, scale, min_mag) with w ~= scale * q - min_mag, which is exactly
        the (d*sc, dmin*m) pair the Q4_K/Q5_K repack helpers expect.
        """
        sub = w.reshape(n_rows, -1, group)
        lo = sub.min(axis=-1)
        hi = sub.max(axis=-1)
        step = np.maximum((hi - lo) / levels, 1e-12)
        q = np.clip(np.rint((sub - lo[..., None]) / step[..., None]), 0, levels)
        return q.astype(np.uint8), step.astype(np.float32), (-lo).astype(np.float32)

    if layout == "ggml_q8_0":
        # block_q8_0 {fp16 d; int8 qs[32]}: symmetric, d = max|w| / 127.
        sub = w.reshape(n_rows, nb, 32)
        d = np.maximum(np.abs(sub).max(axis=-1) / 127.0, 1e-12)
        q = np.clip(np.rint(sub / d[..., None]), -127, 127).astype(np.int8)
        packed = _repack_q8_0(q.reshape(n_rows, k), d.astype(np.float16))
    elif layout == "ggml_q4_K":
        q, sc, mn = _asym(15, 32)
        nib = q.reshape(n_rows, k)
        packed = _repack_q4_k(
            (nib[:, 0::2] | (nib[:, 1::2] << 4)).astype(np.uint8),
            sc.reshape(n_rows, k // 32).astype(np.float16),
            mn.reshape(n_rows, k // 32).astype(np.float16),
        )
    elif layout == "ggml_q5_K":
        q, sc, mn = _asym(31, 32)
        packed = _repack_q5_k(
            q.reshape(n_rows, k),
            sc.reshape(n_rows, k // 32).astype(np.float16),
            mn.reshape(n_rows, k // 32).astype(np.float16),
        )
    else:  # ggml_q6_K — symmetric, w = d * sc * (q - 32), int8 sc per 16 elements
        sub = w.reshape(n_rows, k // 16, 16)
        step16 = np.maximum(np.abs(sub).max(axis=-1) / 32.0, 1e-12)
        per_sb = step16.reshape(n_rows, nb, 16)
        d = np.maximum(per_sb.max(axis=-1) / 127.0, 1e-12)
        sc = np.clip(np.rint(per_sb / d[..., None]), 1, 127).astype(np.int8)
        eff = (d[..., None] * sc.astype(np.float32)).reshape(n_rows, k // 16)
        q = np.clip(np.rint(sub / eff[..., None]) + 32, 0, 63).astype(np.uint8)
        packed = _repack_q6_k(
            q.reshape(n_rows, k), sc.reshape(n_rows, k // 16), d.astype(np.float16)
        )
    return np.ascontiguousarray(packed.reshape(n_rows, k_bytes))


def _gen_byte_buffer(shape: tuple, layout: str, rng: np.random.Generator) -> np.ndarray:
    """Generate a uint8 byte buffer for sentinel/string loops or packed weights.

    ``raw`` — random bytes in [1, 100] (no NUL).
    ``cstrings`` — random non-NUL bytes with NUL terminators sprinkled in plus a
    guaranteed trailing NUL, so the buffer is a run of null-terminated strings that
    always terminates at/before the sentinel `lmt`/`end` pointer.
    ``ggml_q4_K`` / ``ggml_q5_K`` / ``ggml_q6_K`` / ``ggml_q8_0`` — 2-D [N, K_bytes]
    ggml block rows of realistic random quantized weights (see `_gen_ggml_kquant_rows`).
    """
    if layout in _GGML_KQUANT_BLOCK_BYTES:
        return _gen_ggml_kquant_rows(shape, layout, rng)
    n = int(np.prod(shape)) if shape else 0
    buf = rng.integers(1, 101, n, dtype=np.uint8)
    if layout == "cstrings" and n > 0:
        # sprinkle ~1/8 NULs, and force a trailing NUL so strlen never runs past lmt
        nul_idx = rng.random(n) < 0.125
        buf[nul_idx] = 0
        buf[n - 1] = 0
    return buf.reshape(shape)


# ── Axis resolution ────────────────────────────────────────────────────────────

def _input_var_axes(d: Definition) -> set:
    """Var axes that appear in at least one input tensor shape."""
    used: set = set()
    for spec in d.inputs.values():
        if spec.shape is None:
            continue
        for axis in spec.shape:
            ax = d.axes.get(axis)
            if ax is not None and not isinstance(ax, AxisConst):
                used.add(axis)
    return used


def _resolved_axes(d: Definition, w: Workload) -> Dict[str, int]:
    """Merge definition const axes with workload var-axis values.

    Raises if the workload provides an unknown axis, overrides a const, or omits
    a required input var axis.
    """
    out: Dict[str, int] = dict(d.const_axes)
    for name, val in w.axes.items():
        if name not in d.axes:
            raise ValueError(f"Workload axis '{name}' not declared in definition '{d.name}'")
        if isinstance(d.axes[name], AxisConst):
            raise ValueError(
                f"Workload axis '{name}' is const in definition '{d.name}' (value="
                f"{d.const_axes[name]}); workload must not set it"
            )
        out[name] = val
    missing = _input_var_axes(d) - set(out)
    if missing:
        raise ValueError(
            f"Workload missing required input var axes for '{d.name}': {sorted(missing)}"
        )
    return out


# ── dtype mapping ──────────────────────────────────────────────────────────────

_DTYPE_TO_NP = {
    DType.FLOAT64: np.float64,
    DType.FLOAT32: np.float32,
    DType.FLOAT16: np.float16,
    DType.BFLOAT16: ml_dtypes.bfloat16,
    DType.INT64: np.int64,
    DType.INT32: np.int32,
    DType.INT16: np.int16,
    DType.INT8: np.int8,
    DType.UINT64: np.uint64,
    DType.UINT32: np.uint32,
    DType.UINT16: np.uint16,
    DType.UINT8: np.uint8,
    DType.BOOL: np.bool_,
}


def _dtype_to_np(dt: DType):
    np_dt = _DTYPE_TO_NP.get(dt)
    if np_dt is None:
        raise NotImplementedError(f"dtype {dt} not yet supported by inputs.py")
    return np_dt


# ── Main entry point ───────────────────────────────────────────────────────────

_TRACE_ROOT: Optional[Path] = None


def set_trace_root(root) -> None:
    """Record the warehouse root so `{"type": "tensor"}` workload inputs resolve.

    TraceSet.from_path calls this; ARMBENCH_TRACE_ROOT overrides it for callers that
    build a Definition/Workload pair by hand (the e2e scripts do).
    """
    global _TRACE_ROOT
    _TRACE_ROOT = Path(root)


def _trace_root() -> Path:
    env = os.environ.get("ARMBENCH_TRACE_ROOT")
    if env:
        return Path(env)
    if _TRACE_ROOT is not None:
        return _TRACE_ROOT
    return Path(__file__).resolve().parent.parent.parent / "bench-trace"


def _load_tensor_input(name: str, rel: str, shape, np_dtype) -> np.ndarray:
    root = _trace_root()
    path = (root / rel).resolve()
    if root.resolve() not in path.parents and path != root.resolve():
        raise ValueError(f"tensor input '{name}' path escapes the trace root: {rel}")
    if not path.exists():
        raise FileNotFoundError(
            f"tensor input '{name}' missing: {path} (regenerate with "
            f"scripts/e2e/dump_to_workloads.py, or re-pull bench-trace)"
        )
    arr = np.load(path)
    if tuple(arr.shape) != tuple(shape):
        raise ValueError(
            f"tensor input '{name}' has shape {tuple(arr.shape)}, workload axes need {tuple(shape)}"
        )
    if arr.dtype != np_dtype:
        arr = arr.astype(np_dtype)
    return arr


def gen_inputs_for_workload(d: Definition, w: Workload) -> Dict[str, object]:
    """Build the input dict for `Definition.reference.run(**inputs)`.

    Every entry in `d.inputs` must have a corresponding entry in `w.inputs`:
    - `{"type": "random"}` → numpy array generated from uuid-seeded rng
    - `{"type": "scalar", "value": v}` → Python scalar (int / float / bool)
    - `{"type": "tensor", "path": p}` → real tensor loaded from the trace warehouse

    Raises ValueError if any definition input is absent from the workload.
    """
    axes = _resolved_axes(d, w)
    rng = np.random.default_rng(_uuid_to_seed(w.uuid))
    out: Dict[str, object] = {}

    # q8_0 block quantization (int8 data + fp16 `*_scales`): generate the way real
    # quantization does — signed/zero-centered int8 with small scales (~max|x|/127).
    # All-positive int8 * O(1) scales has no cancellation and explodes through wide
    # matmuls (MoE chains two, reaching ~1e16 → meaningless), whereas signed data +
    # small scales keeps dequantize→matmul in range. Scoped to q8_0 definitions;
    # leaves loop_* integer inputs, Q4_K (uint8 data), and w8a8ch (fp32 scales) as-is.
    _q8 = (
        any(s.dtype == DType.INT8 for s in d.inputs.values())
        and any(
            n.endswith("_scales") and s.dtype == DType.FLOAT16
            for n, s in d.inputs.items()
        )
    )

    for tname, tspec in d.inputs.items():
        wi = w.inputs.get(tname)
        if wi is None:
            raise ValueError(
                f"Workload '{w.uuid}' is missing input '{tname}' "
                f"(definition '{d.name}' requires it)"
            )
        if wi.type == "scalar":
            out[tname] = wi.value
            continue
        if tspec.shape is None:
            raise ValueError(
                f"Definition '{d.name}' input '{tname}' has shape=null "
                f"but workload declares it as random/bytes (expected scalar)"
            )
        shape = tuple(axes[a] for a in tspec.shape)
        if wi.type == "bytes":
            out[tname] = _gen_byte_buffer(shape, wi.layout, rng)
            continue
        if wi.type == "tensor":
            out[tname] = _load_tensor_input(tname, wi.path, shape, _dtype_to_np(tspec.dtype))
            continue
        # type == "random"
        if _q8 and tspec.dtype == DType.INT8:
            # signed/zero-centered int8 quants, like real q8_0
            out[tname] = rng.integers(-127, 128, shape).astype(np.int8)
        elif _q8 and tspec.dtype == DType.FLOAT16 and tname.endswith("_scales"):
            # realistic per-block scales ~ max|x|/127 (keeps dequant values O(1))
            out[tname] = rng.uniform(1.0 / 255.0, 1.0 / 64.0, shape).astype(np.float16)
        else:
            non_negative = tname.endswith(("_scales", "_mins"))
            out[tname] = _gen_random_tensor(
                shape, _dtype_to_np(tspec.dtype), rng, non_negative=non_negative
            )

    return out


def shape_of(arr_or_scalar: object) -> Tuple[int, ...]:
    """Best-effort shape extraction for either numpy arrays or python scalars."""
    if hasattr(arr_or_scalar, "shape"):
        return tuple(arr_or_scalar.shape)  # type: ignore[attr-defined]
    return ()


__all__ = [
    "make_weights",
    "make_mat_ramp",
    "make_mat_ramp_2d",
    "gen_inputs_for_workload",
    "shape_of",
]
