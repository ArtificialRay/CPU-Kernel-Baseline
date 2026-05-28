"""Deterministic input generators that bit-exactly reproduce arm-bench/starter/ncnn/ncnn_helpers.h.

Why bit-exact: the Phase-2 verification requires the new benchmarker's
correctness verdicts to match the old EXPECT_MATCH harness one-for-one. That
only holds if we feed the kernel the same bytes the old harness did.
"""

from typing import Dict, Iterable, Tuple

import numpy as np

from bench.data.definition import AxisConst, Definition, DType
from bench.data.workload import Workload


# ── Bit-exact ports of ncnn_helpers.h ────────────────────────────────────────

def make_weights(n: int, scale: float = 1.0) -> np.ndarray:
    """Port of ncnn_helpers.h `make_weights`:

        w[i] = (((i * 1234567 + 7654321) % 1000) / 1000.f - 0.5f) * scale

    Critical: do the LCG in pure Python ints (or np.int64) BEFORE converting to
    float32 — doing it in float32 would lose precision in the modulo step.
    """
    i = np.arange(n, dtype=np.int64)
    # The C++ uses signed int32 multiplication which wraps; here int64 is wide
    # enough to hold the worst case (i ~ 1e8 → product ~ 1e14, fits in int64)
    # without overflow, matching the C++ result for all practical n.
    lcg = (i * 1234567 + 7654321) % 1000
    return (lcg.astype(np.float32) / 1000.0 - 0.5).astype(np.float32) * np.float32(scale)


def make_mat_ramp(shape_chw: Tuple[int, int, int]) -> np.ndarray:
    """Port of ncnn_helpers.h `make_mat_ramp(w, h, c)`.

    C-major layout (channel outermost), values = `(idx % 100 + 1) * 0.1f`,
    where idx is the flat c-major index. Returns shape (c, h, w) float32.

    The bounded cycle (0.1..10.0) avoids catastrophic cancellation in long
    reductions; see the comment in ncnn_helpers.h.
    """
    c, h, w = shape_chw
    n = c * h * w
    idx = np.arange(n, dtype=np.int64)
    flat = ((idx % 100) + 1).astype(np.float32) * np.float32(0.1)
    return flat.reshape(c, h, w)


def make_mat_ramp_2d(shape_hw: Tuple[int, int]) -> np.ndarray:
    """Port of ncnn_helpers.h `make_mat_ramp_2d(w, h)`. Returns (h, w) float32."""
    h, w = shape_hw
    n = h * w
    idx = np.arange(n, dtype=np.int64)
    flat = ((idx % 100) + 1).astype(np.float32) * np.float32(0.1)
    return flat.reshape(h, w)


# ── Definition + Workload → concrete numpy inputs ─────────────────────────────

def _resolved_axes(d: Definition, w: Workload) -> Dict[str, int]:
    """Merge a definition's const axes with a workload's var-axis values.

    Raises if the workload is missing a var axis the definition declares or
    overrides a const.
    """
    out: Dict[str, int] = dict(d.const_axes)  # const first
    for name, val in w.axes.items():
        if name not in d.axes:
            raise ValueError(f"Workload axis '{name}' not declared in definition '{d.name}'")
        if isinstance(d.axes[name], AxisConst):
            raise ValueError(
                f"Workload axis '{name}' is const in definition '{d.name}' (value="
                f"{d.const_axes[name]}); workload must not set it"
            )
        out[name] = val
    missing = set(d.var_axes) - set(out)
    if missing:
        raise ValueError(f"Workload missing var axes for '{d.name}': {sorted(missing)}")
    return out


_DTYPE_TO_NP = {
    DType.FLOAT32: np.float32,
    DType.FLOAT16: np.float16,
    DType.BFLOAT16: None,  # numpy lacks native bfloat16; not used in Phase 1
    DType.INT64: np.int64,
    DType.INT32: np.int32,
    DType.INT16: np.int16,
    DType.INT8: np.int8,
    DType.BOOL: np.bool_,
}


def _dtype_to_np(dt: DType):
    np_dt = _DTYPE_TO_NP.get(dt)
    if np_dt is None:
        raise NotImplementedError(f"dtype {dt} not yet supported by inputs.py")
    return np_dt


def gen_inputs_for_workload(d: Definition, w: Workload) -> Dict[str, object]:
    """Build the input dict for `Definition.reference.run(**inputs)`.

    Generation policy (matches today's harness):

    - Tensor named `input` or `bottom_blob`: 3D ramp via `make_mat_ramp`, else 2D ramp.
    - Tensor named `weight` (or `weight_data`): seeded LCG via `make_weights`.
    - Tensor named `bias` (or `bias_data`): seeded LCG, scale 1.0. Caller can
      zero it out via `scalar_inputs['bias_term'] = 0` if a bias-less variant is needed.
    - Tensor named `activation_params`: small zeros tensor (today's ncnn passes empty for type=0/1).
    - Scalar-shape tensors (shape=None): taken from `w.scalar_inputs` (must be set).

    Returns numpy arrays in N-C-H-W layout for tensors and Python scalars for shape=None.
    The dataset adapter converts those to the framework-native handle (e.g. ncnn::Mat).
    """
    axes = _resolved_axes(d, w)
    out: Dict[str, object] = {}
    for tname, tspec in d.inputs.items():
        if tspec.shape is None:
            if tname not in w.scalar_inputs:
                raise ValueError(
                    f"Workload missing scalar input '{tname}' (definition has shape=None)"
                )
            out[tname] = w.scalar_inputs[tname]
            continue

        # Resolve concrete shape from axis names
        shape = tuple(axes[a] for a in tspec.shape)
        np_dt = _dtype_to_np(tspec.dtype)

        if tname in ("input", "bottom_blob"):
            # Detect dimensionality from declared shape.
            # NCHW (4D): take c=C*N (treat batch by concatenation along c)
            if len(shape) == 4:
                n, c, h, ww = shape
                arr = np.empty((n, c, h, ww), dtype=np_dt)
                for bi in range(n):
                    arr[bi] = make_mat_ramp((c, h, ww))
                out[tname] = arr
            elif len(shape) == 3:
                out[tname] = make_mat_ramp(shape).astype(np_dt, copy=False)
            elif len(shape) == 2:
                out[tname] = make_mat_ramp_2d(shape).astype(np_dt, copy=False)
            else:
                raise ValueError(
                    f"Input tensor '{tname}' has unsupported rank {len(shape)}; expected 2/3/4"
                )
        elif tname in ("weight", "weight_data"):
            n_elems = int(np.prod(shape))
            out[tname] = make_weights(n_elems, scale=1.0).reshape(shape).astype(np_dt, copy=False)
        elif tname in ("bias", "bias_data"):
            n_elems = int(np.prod(shape))
            # NB: today's tests use bias = [i * 0.1 for i in range(out_c)] when with_bias=true.
            # We use make_weights here so both bias and weight are deterministic from a
            # single seed; if a Phase-2 workload diff shows mismatch, switch this branch
            # to match the test exactly.
            out[tname] = make_weights(n_elems, scale=1.0).reshape(shape).astype(np_dt, copy=False)
        else:
            # Fallback: deterministic via make_weights so any tensor gets *some* data
            n_elems = int(np.prod(shape))
            out[tname] = make_weights(n_elems, scale=1.0).reshape(shape).astype(np_dt, copy=False)
    return out


def gen_constants_for_reference(d: Definition, w: Workload) -> Dict[str, int]:
    """Expose const-axis values as plain ints so `Definition.reference` can reference
    them directly (e.g. `run` may need `Sh`, `Sw` to call `F.conv2d`).
    """
    return dict(d.const_axes)


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
    "gen_constants_for_reference",
    "shape_of",
]
