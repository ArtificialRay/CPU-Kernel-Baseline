"""Input generators for benchmark workloads.

`gen_inputs_for_workload` is the main entry point — it reads each input's type
from `workload.inputs` and either returns the scalar value directly or generates
a bounded-uniform random tensor seeded from the workload's uuid.

The legacy `make_weights` / `make_mat_ramp` / `make_mat_ramp_2d` functions are
kept for standalone testing but are no longer called by the harness.
"""

import hashlib
from typing import Dict, Iterable, Tuple

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


def _gen_random_tensor(shape: tuple, dtype, rng: np.random.Generator) -> np.ndarray:
    """Generate a bounded random tensor suitable for kernel benchmarking.

    Float tensors use uniform(-1, 1) to avoid catastrophic cancellation in
    long reductions; integer tensors use integers in [1, 100].
    """
    if np.issubdtype(dtype, np.floating):
        return rng.uniform(-1.0, 1.0, shape).astype(dtype)
    elif np.issubdtype(dtype, np.integer):
        return rng.integers(1, 101, shape).astype(dtype)
    else:  # bool
        return rng.integers(0, 2, shape, dtype=np.uint8).astype(np.bool_)


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
    DType.BFLOAT16: None,  # numpy lacks native bfloat16
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

def gen_inputs_for_workload(d: Definition, w: Workload) -> Dict[str, object]:
    """Build the input dict for `Definition.reference.run(**inputs)`.

    Every entry in `d.inputs` must have a corresponding entry in `w.inputs`:
    - `{"type": "random"}` → numpy array generated from uuid-seeded rng
    - `{"type": "scalar", "value": v}` → Python scalar (int / float / bool)

    Raises ValueError if any definition input is absent from the workload.
    """
    axes = _resolved_axes(d, w)
    rng = np.random.default_rng(_uuid_to_seed(w.uuid))
    out: Dict[str, object] = {}

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
        # type == "random"
        if tspec.shape is None:
            raise ValueError(
                f"Definition '{d.name}' input '{tname}' has shape=null "
                f"but workload declares it as random (expected scalar)"
            )
        shape = tuple(axes[a] for a in tspec.shape)
        out[tname] = _gen_random_tensor(shape, _dtype_to_np(tspec.dtype), rng)

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
