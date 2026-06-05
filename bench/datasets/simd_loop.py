"""simd-loop dataset adapter: numpy ↔ flat C arrays via ctypes.

Mirrors bench/datasets/raw.py's pattern but for the simd-loop baseline ABI.
Each loop_NNN problem has a per-op struct (loop_NNN_data) with two pointer
fields, an int length, and a scalar result. The harness shim (in
bench/compile/builders/simd_loop_harness/<op>.cpp) fills that struct and calls
inner_loop_NNN(&data). The entry point the runner binds is:

    int armbench_entry_loop_NNN(void *a, void *b, int64_t n, void *res_out)

SIGNATURES is the load-bearing mirror of those C signatures; edit one, edit
the other.

To add a new loop:
1. Add simd_loop_harness/loop_NNN.{h,cpp}.
2. Add an entry to SIGNATURES (arg order must match the C signature).
3. Add an entry to _RESULT_DTYPE (numpy dtype of the scalar result).
4. Add a Definition JSON and Workload JSONL under bench-trace/.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

_C_VOID_P = ctypes.c_void_p
_C_INT64 = ctypes.c_int64

# All four basic inner-product loops share the same flat signature:
#   (void* a, void* b, int64_t n, void* res_out)
SIGNATURES: Dict[str, List[type]] = {
    "loop_001": [_C_VOID_P, _C_VOID_P, _C_INT64, _C_VOID_P],
    "loop_002": [_C_VOID_P, _C_VOID_P, _C_INT64, _C_VOID_P],
    "loop_003": [_C_VOID_P, _C_VOID_P, _C_INT64, _C_VOID_P],
    "loop_004": [_C_VOID_P, _C_VOID_P, _C_INT64, _C_VOID_P],
}

# Numpy dtype for the scalar result written by the harness into res_out.
_RESULT_DTYPE: Dict[str, type] = {
    "loop_001": np.float32,
    "loop_002": np.uint32,
    "loop_003": np.float64,
    "loop_004": np.uint64,
}


@dataclass
class SimdLoopContext:
    entry_args: Tuple[Any, ...]
    # Contiguous numpy arrays backing the void* pointers — kept alive for
    # the duration of the kernel call.
    _arrays: list
    # 1-element numpy array where the harness writes the scalar result.
    res_buf: np.ndarray


class SimdLoopDataset:
    """Adapter for simd-loop baseline kernels.

    Protocol (same as NcnnDataset / RawDataset):
        ctx = ds.wrap_inputs(np_inputs, scalar_args, op_type, lib)
        entry(*ctx.entry_args)
        out = ds.unwrap_output(ctx)
        ds.release(ctx)
    """

    name = "simd-loop"

    def wrap_inputs(
        self,
        np_inputs: Dict[str, np.ndarray],
        scalar_args: Dict[str, int],
        op_type: str,
        lib: ctypes.CDLL,
    ) -> SimdLoopContext:
        if op_type not in SIGNATURES:
            raise NotImplementedError(f"SimdLoopDataset: op_type '{op_type}' not registered")
        a = np.ascontiguousarray(np_inputs["a"])
        b = np.ascontiguousarray(np_inputs["b"])
        n = int(scalar_args["N"])
        res_buf = np.zeros(1, dtype=_RESULT_DTYPE[op_type])
        entry_args = (
            ctypes.cast(a.ctypes.data, _C_VOID_P),
            ctypes.cast(b.ctypes.data, _C_VOID_P),
            _C_INT64(n),
            ctypes.cast(res_buf.ctypes.data, _C_VOID_P),
        )
        return SimdLoopContext(entry_args=entry_args, _arrays=[a, b], res_buf=res_buf)

    def unwrap_output(self, ctx: SimdLoopContext) -> np.ndarray:
        return ctx.res_buf.copy()

    def release(self, ctx: SimdLoopContext) -> None:
        ctx._arrays.clear()
