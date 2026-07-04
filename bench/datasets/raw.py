"""raw dataset adapter: numpy ↔ bare `float*` via ctypes.

Used for every CANDIDATE solution regardless of its declared `dataset`. The
candidate is built by CandidateBuilder with no framework dependency, so its
`armbench_entry_<op>` takes plain float pointers + the truly runtime dims.
Const dims (Cout, Kh, Kw, …, pad, activation) are baked as constexpr into
each candidate's kernel.cpp
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from bench.data.definition import Definition

_C_FLOAT_P = ctypes.POINTER(ctypes.c_float)


@dataclass
class RawContext:
    """Holds buffers alive between wrap and unwrap so ctypes pointers stay valid."""

    entry_args: Tuple[Any, ...]
    output: np.ndarray
    _keepalive: List[np.ndarray] = field(default_factory=list)


def _compute_output_shape(definition: Definition, np_inputs: Dict[str, Any]) -> Tuple[int, ...]:
    """Derive the output tensor shape from definition axes + constraints.

    Builds a namespace from const axes, var dims (extracted from input tensor
    shapes), and scalar inputs (pad_top, pad_left, …), then eval()s each
    constraint expression to get derived axes (H_out, W_out, …).

    Works for operators with no constraints too (batchnorm, gemm): once step 2
    populates ns from input shapes, output axes are already present.
    """
    ns: Dict[str, Any] = dict(definition.const_axes)
    for tname, tspec in definition.inputs.items():
        if tspec.shape is not None and tname in np_inputs:
            for ax, val in zip(tspec.shape, np_inputs[tname].shape):
                ns.setdefault(ax, val)
        elif tspec.shape is None and tname in np_inputs:
            ns[tname] = int(np_inputs[tname])
    for constraint in definition.constraints:
        lhs, _, rhs = constraint.partition("==")
        ns[lhs.strip()] = int(eval(rhs.strip(), {"__builtins__": {}}, ns))  # noqa: S307
    out_spec = next(iter(definition.outputs.values()))
    return tuple(ns[ax] for ax in out_spec.shape)


class RawDataset:
    """Adapter for the slim raw `float*` candidate ABI.

    Each candidate kernel bakes its const dims as constexpr at codegen time,
    so the runtime entry takes ONLY pointers + the truly var dims (N, H, W for
    conv2d). entry_args carries fully-typed ctypes objects — no fn.argtypes
    needed in the runner.

    ABI convention (matches gen_candidate_bindings.py output):
        armbench_entry_<op>(float* primary, float* output,
                            float* rest...,
                            int var_dim_0, int var_dim_1, ...)
    Tensor order follows definition.inputs (non-scalars); var dims are taken
    from the primary input's shape template in definition order.
    """

    name = "raw"

    def __init__(self) -> None:
        pass

    def wrap_inputs(
        self,
        np_inputs: Dict[str, Any],
        op_type: str,
        lib: ctypes.CDLL,
        *,
        definition: Definition,
        out_shape: Optional[Tuple[int, ...]] = None,
    ) -> RawContext:
        if out_shape is None:
            out_shape = _compute_output_shape(definition, np_inputs)

        # Tensor inputs in definition order; scalars (shape is None) are skipped
        tensor_specs = [(n, s) for n, s in definition.inputs.items() if s.shape is not None]

        arrays: List[Optional[np.ndarray]] = [
            np.ascontiguousarray(np_inputs[n], dtype=np.float32)
            if np_inputs.get(n) is not None else None
            for n, _ in tensor_specs
        ]

        output = np.zeros(out_shape, dtype=np.float32)

        def _fptr(a: Optional[np.ndarray]) -> Any:
            if a is None:
                return ctypes.cast(None, _C_FLOAT_P)
            return a.ctypes.data_as(_C_FLOAT_P)

        # Var dims: scan every tensor's shape template (not just primary), dedup
        # by axis name so an axis seen in an earlier tensor isn't repeated.
        seen_axes: set = set()
        var_dim_args = []
        for arr, (_, spec) in zip(arrays, tensor_specs):
            if arr is None:
                continue
            for i, ax in enumerate(spec.shape):
                if definition.axes[ax].type == "var" and ax not in seen_axes:
                    seen_axes.add(ax)
                    var_dim_args.append(ctypes.c_int(arr.shape[i]))

        entry_args: Tuple[Any, ...] = (
            _fptr(arrays[0]),
            output.ctypes.data_as(_C_FLOAT_P),
            *(_fptr(a) for a in arrays[1:]),
            *var_dim_args,
        )

        keepalive = [a for a in arrays if a is not None] + [output]
        return RawContext(entry_args=entry_args, output=output, _keepalive=keepalive)

    def unwrap_output(self, ctx: RawContext) -> np.ndarray:
        return ctx.output

    def release(self, ctx: RawContext) -> None:
        ctx._keepalive.clear()
