"""simd-loop dataset adapter: numpy ↔ flat C arrays via ctypes.

Calling convention for every loop:
    int armbench_entry_loop_NNN(void* in1, [void* in2, ...], [void* scratch, ...], int64_t n, void* res_out)

For scalar-output loops:  res_out → single scalar value written by the kernel.
For array-output loops:   res_out → pre-allocated array of N elements (the output buffer).
For inplace-sort loops:   res_out is unused; the first ptr (data) is sorted in-place.

Adapter metadata is read from Definition.simd_loop_meta (set by gen_simd_loop_harness.py).
No hard-coded per-loop registries — just the definition JSON.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from bench.data.definition import Definition

_C_VOID_P = ctypes.c_void_p
_C_INT64   = ctypes.c_int64

_DTYPE_MAP: dict[str, Any] = {
    "float32": np.float32, "float64": np.float64,
    "int64":   np.int64,   "int32":   np.int32,
    "int16":   np.int16,   "int8":    np.int8,
    "uint64":  np.uint64,  "uint32":  np.uint32,
}


# ── Context ───────────────────────────────────────────────────────────────────

@dataclass
class SimdLoopContext:
    entry_args: Tuple[Any, ...]
    _arrays:    list          # all live arrays (inputs + scratch + res_buf for inplace)
    res_buf:    np.ndarray    # result buffer: scalar (1-elem), array (N-elem), or inplace ref
    _n:         int = 0       # user-visible output length; 0 → use full res_buf


# ── Adapter ───────────────────────────────────────────────────────────────────

class SimdLoopDataset:
    """Adapter for simd-loop baseline kernels.

    Protocol (same as NcnnDataset / RawDataset):
        ctx = ds.wrap_inputs(np_inputs, op_type, lib, definition=d)
        entry(*ctx.entry_args)
        out = ds.unwrap_output(ctx)
        ds.release(ctx)
    """

    name = "simd-loop"

    def wrap_inputs(
        self,
        np_inputs: Dict[str, np.ndarray],
        op_type: str,
        lib: ctypes.CDLL,
        *,
        definition: Definition,
        out_shape: Optional[Tuple[int, ...]] = None,
    ) -> SimdLoopContext:
        if definition.simd_loop_meta is None:
            raise NotImplementedError(
                f"SimdLoopDataset: definition for {op_type!r} missing simd_loop_meta. "
                "Re-run scripts/gen_simd_loop_harness.py to regenerate definitions."
            )
        meta = definition.simd_loop_meta
        pad = int(meta.array_pad)

        # Derive output dtype from Definition.outputs
        out_spec = next(iter(definition.outputs.values()))
        result_dtype = _DTYPE_MAP[out_spec.dtype.value]
        output_is_array = out_spec.shape is not None

        if meta.output_inplace:
            # In-place: the first input is sorted in-place; scratch bufs follow.
            input_names = list(definition.inputs.keys())
            data_arr = np.ascontiguousarray(np_inputs[input_names[0]].copy())
            n = int(data_arr.shape[0])
            scratch = [np.zeros(n, dtype=_DTYPE_MAP[s.dtype]) for s in meta.scratch]
            all_arrs = [data_arr] + scratch
            ptrs = [ctypes.cast(a.ctypes.data, _C_VOID_P) for a in all_arrs]
            dummy = np.zeros(1, dtype=np.int64)
            entry_args = (
                tuple(ptrs)
                + (_C_INT64(n),)
                + (ctypes.cast(dummy.ctypes.data, _C_VOID_P),)
            )
            return SimdLoopContext(entry_args=entry_args, _arrays=all_arrs + [dummy],
                                   res_buf=data_arr, _n=0)

        arrays = [np.ascontiguousarray(np_inputs[name]) for name in definition.inputs]
        n = int(arrays[0].shape[0])
        if output_is_array:
            if pad > 0:
                arrays = [
                    np.concatenate([a, np.zeros(pad, dtype=a.dtype)])
                    for a in arrays
                ]
            res_buf = np.zeros(n + pad, dtype=result_dtype)
        else:
            res_buf = np.zeros(1, dtype=result_dtype)
        ptrs = [ctypes.cast(a.ctypes.data, _C_VOID_P) for a in arrays]
        entry_args = tuple(ptrs) + (_C_INT64(n),) + (ctypes.cast(res_buf.ctypes.data, _C_VOID_P),)
        return SimdLoopContext(entry_args=entry_args, _arrays=arrays, res_buf=res_buf,
                               _n=n if pad > 0 else 0)

    def unwrap_output(self, ctx: SimdLoopContext) -> np.ndarray:
        arr = ctx.res_buf.copy()
        if ctx._n > 0 and len(arr) > ctx._n:
            arr = arr[:ctx._n]
        return arr

    def release(self, ctx: SimdLoopContext) -> None:
        ctx._arrays.clear()
