"""raw dataset adapter: numpy ↔ bare `float*` via ctypes.

Used for every CANDIDATE solution regardless of its declared `dataset`. The
candidate is built by CandidateBuilder with no framework dependency, so its
`armbench_entry_<op>` takes plain float pointers + explicit shape ints (see
bench/compile/builders/candidate_harness/<op>.h).

SIGNATURES here is the load-bearing mirror of that C-ABI contract; the runner
binds the entry with these argtypes when running a candidate. Edit one, edit
the other.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

_C_FLOAT_P = ctypes.POINTER(ctypes.c_float)
_C_INT = ctypes.c_int

# Mirrors armbench_entry_conv2d in candidate_harness/conv2d.h. Order is exact:
#   const float* input, float* output, const float* weight, const float* bias,
#   int N, int Cin, int H, int W,
#   int Cout, int Kh, int Kw, int Sh, int Sw, int Dh, int Dw,
#   int pad_top, int pad_left,
#   int activation_type, const float* act_params, int n_act
SIGNATURES: Dict[str, List[type]] = {
    "conv2d": [
        _C_FLOAT_P, _C_FLOAT_P, _C_FLOAT_P, _C_FLOAT_P,
        _C_INT, _C_INT, _C_INT, _C_INT,
        _C_INT, _C_INT, _C_INT, _C_INT, _C_INT, _C_INT, _C_INT,
        _C_INT, _C_INT,
        _C_INT, _C_FLOAT_P, _C_INT,
    ],
}


def _out_dim(in_dim: int, pad: int, k: int, dil: int, stride: int) -> int:
    ext_k = dil * (k - 1) + 1
    return (in_dim + 2 * pad - ext_k) // stride + 1


@dataclass
class RawContext:
    """Holds buffers alive between wrap and unwrap so ctypes pointers stay valid."""

    entry_args: Tuple[Any, ...]
    output: np.ndarray
    # Keep input/weight/act buffers referenced for the duration of the call.
    _keepalive: List[np.ndarray] = field(default_factory=list)


class RawDataset:
    """Adapter for the raw `float*` candidate ABI."""

    name = "raw"

    def __init__(self) -> None:
        pass

    def wrap_inputs(
        self,
        np_inputs: Dict[str, Any],
        scalar_args: Dict[str, int],
        op_type: str,
        lib: ctypes.CDLL,
    ) -> RawContext:
        if op_type != "conv2d":
            raise NotImplementedError(f"RawDataset: op_type '{op_type}' not yet supported")

        inp = np.ascontiguousarray(np_inputs["input"], dtype=np.float32)
        # Input is NCHW (4D) per the conv2d definitions; tolerate 3D (treat N=1).
        if inp.ndim == 4:
            N, Cin, H, W = inp.shape
        elif inp.ndim == 3:
            N = 1
            Cin, H, W = inp.shape
        else:
            raise ValueError(f"RawDataset conv2d: input must be 3D/4D, got {inp.shape}")

        weight = np.ascontiguousarray(np_inputs["weight"], dtype=np.float32)

        Cout = int(scalar_args["out_c"])
        Kh = int(scalar_args["kernel_h"])
        Kw = int(scalar_args["kernel_w"])
        Sh = int(scalar_args["stride_h"])
        Sw = int(scalar_args["stride_w"])
        Dh = int(scalar_args["dilation_h"])
        Dw = int(scalar_args["dilation_w"])
        pad_top = int(scalar_args["pad_top"])
        pad_left = int(scalar_args["pad_left"])
        activation_type = int(scalar_args["activation_type"])

        H_out = _out_dim(H, pad_top, Kh, Dh, Sh)
        W_out = _out_dim(W, pad_left, Kw, Dw, Sw)

        # bias: the conv2d reference never adds bias (F.conv2d(..., None, ...)),
        # so we pass NULL — correctness must not depend on it.
        bias_arr = np_inputs.get("bias")
        if bias_arr is not None:
            bias_arr = np.ascontiguousarray(bias_arr, dtype=np.float32)
            bias_ptr = bias_arr.ctypes.data_as(_C_FLOAT_P)
        else:
            bias_ptr = None  # → NULL

        act_arr = np_inputs.get("activation_params")
        if act_arr is not None and np.asarray(act_arr).size > 0:
            act_arr = np.ascontiguousarray(act_arr, dtype=np.float32)
            act_ptr = act_arr.ctypes.data_as(_C_FLOAT_P)
            n_act = int(act_arr.size)
        else:
            act_arr = None
            act_ptr = None
            n_act = 0

        output = np.zeros((N, Cout, H_out, W_out), dtype=np.float32)

        entry_args: Tuple[Any, ...] = (
            inp.ctypes.data_as(_C_FLOAT_P),
            output.ctypes.data_as(_C_FLOAT_P),
            weight.ctypes.data_as(_C_FLOAT_P),
            bias_ptr,
            N, Cin, H, W,
            Cout, Kh, Kw, Sh, Sw, Dh, Dw,
            pad_top, pad_left,
            activation_type, act_ptr, n_act,
        )

        keepalive = [inp, weight, output]
        if bias_arr is not None:
            keepalive.append(bias_arr)
        if act_arr is not None:
            keepalive.append(act_arr)

        return RawContext(entry_args=entry_args, output=output, _keepalive=keepalive)

    def unwrap_output(self, ctx: RawContext) -> np.ndarray:
        # Already host memory written in place by the kernel; no copy-back needed.
        return ctx.output

    def release(self, ctx: RawContext) -> None:
        ctx._keepalive.clear()
