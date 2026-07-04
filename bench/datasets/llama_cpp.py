"""llama.cpp dataset adapter: numpy ↔ ggml-ready buffers via ctypes.

Every llama.cpp baseline ships a self-contained binding.cpp (see
`scripts/gen_llamacpp_baseline_solution.py`) whose entry has one generic ABI:

    int armbench_entry_<op_type>(const void* const* inputs,
                                 void*              output,
                                 const int64_t*     var_axes);

- `inputs`  — one pointer per Definition input tensor, in definition order.
  fp32 tensors are passed as contiguous float32 buffers. Q8_0 tensor pairs
  (an int8 quant tensor immediately followed by its float16 scales tensor,
  e.g. `A`+`A_scales`, `gate_proj`+`gate_scales`) are repacked here into
  ggml block_q8_0 layout — 34-byte blocks of {fp16 d; int8 qs[32]} along the
  last axis — and passed at the int8 tensor's slot; the consumed scales slot
  carries NULL. Const axes are baked as constexpr in the binding, so only
  buffers cross the ABI.
- `output`  — contiguous float32 buffer, allocated here from the reference
  output shape (ggml's [ne0..] reversed order equals numpy row-major).
- `var_axes` — values of the Definition's var axes, in declaration order
  (e.g. [M] for gemm, [M, S] for mha, [n_tokens] for moe).
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from bench.data.definition import Definition, DType

_Q8_0_BLOCK = 32
_Q8_0_BLOCK_BYTES = 34  # sizeof(block_q8_0) = 2 (fp16 d) + 32 (int8 qs)


def _repack_q8_0(qs: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Repack (int8 quants, fp16 scales) into ggml block_q8_0 rows.

    qs [..., K] int8 + scales [..., K/32] float16 → uint8 [..., K/32, 34]
    where each 34-byte block is {fp16 d (little-endian); int8 qs[32]}.
    """
    *lead, k = qs.shape
    if k % _Q8_0_BLOCK != 0:
        raise ValueError(f"q8_0 tensor last dim {k} not a multiple of {_Q8_0_BLOCK}")
    nblk = k // _Q8_0_BLOCK
    if tuple(scales.shape) != (*lead, nblk):
        raise ValueError(
            f"q8_0 scales shape {scales.shape} does not match quants shape "
            f"{qs.shape} (expected {(*lead, nblk)})"
        )
    rows = int(np.prod(lead)) if lead else 1
    q = np.ascontiguousarray(qs, dtype=np.int8).reshape(rows, nblk, _Q8_0_BLOCK)
    s = np.ascontiguousarray(scales, dtype=np.dtype("<f2")).reshape(rows, nblk)
    buf = np.empty((rows, nblk, _Q8_0_BLOCK_BYTES), dtype=np.uint8)
    buf[:, :, 0:2] = s.view(np.uint8).reshape(rows, nblk, 2)
    buf[:, :, 2:] = q.view(np.uint8)
    return buf


@dataclass
class LlamaCppContext:
    """Holds buffers alive between wrap and unwrap so ctypes pointers stay valid."""

    entry_args: Tuple[Any, ...]
    output: np.ndarray
    _keepalive: List[np.ndarray] = field(default_factory=list)


class LlamaCppDataset:
    """Adapter for the llama.cpp (ggml) baseline ABI.

    Usage mirrors the other adapters:
        ds = LlamaCppDataset()
        ctx = ds.wrap_inputs(np_inputs, op_type, lib, definition=definition,
                             out_shape=ref.shape)
        ret = entry(*ctx.entry_args)
        out = ds.unwrap_output(ctx)
        ds.release(ctx)
    """

    name = "llama.cpp"

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
    ) -> LlamaCppContext:
        if out_shape is None:
            raise ValueError(
                "LlamaCppDataset requires out_shape (the reference output shape)"
            )

        tensor_specs = [
            (n, s) for n, s in definition.inputs.items() if s.shape is not None
        ]

        keepalive: List[np.ndarray] = []
        ptrs: List[Optional[int]] = []
        i = 0
        while i < len(tensor_specs):
            name, spec = tensor_specs[i]
            arr = np_inputs.get(name)
            if arr is None:
                raise ValueError(f"Missing input tensor '{name}' for '{definition.name}'")
            nxt = tensor_specs[i + 1] if i + 1 < len(tensor_specs) else None
            if (
                spec.dtype == DType.INT8
                and nxt is not None
                and nxt[1].dtype == DType.FLOAT16
            ):
                # q8_0 pair: quants + scales → one repacked block_q8_0 buffer at
                # the quants slot, NULL at the scales slot.
                scales = np_inputs.get(nxt[0])
                if scales is None:
                    raise ValueError(
                        f"Missing q8_0 scales tensor '{nxt[0]}' for '{definition.name}'"
                    )
                packed = _repack_q8_0(np.asarray(arr), np.asarray(scales))
                keepalive.append(packed)
                ptrs.append(packed.ctypes.data)
                ptrs.append(None)
                i += 2
                continue
            contiguous = np.ascontiguousarray(arr, dtype=np.float32)
            keepalive.append(contiguous)
            ptrs.append(contiguous.ctypes.data)
            i += 1

        output = np.zeros(out_shape, dtype=np.float32)
        keepalive.append(output)

        # Var axis values, in definition.axes declaration order, resolved from
        # the input tensor shapes.
        ns: Dict[str, int] = {}
        for tname, tspec in definition.inputs.items():
            if tspec.shape is not None and tname in np_inputs:
                for ax, val in zip(tspec.shape, np.asarray(np_inputs[tname]).shape):
                    ns.setdefault(ax, int(val))
        var_vals = [ns[a] for a in definition.var_axes]
        var_axes_arr = (ctypes.c_int64 * max(len(var_vals), 1))(*var_vals)

        inputs_arr = (ctypes.c_void_p * len(ptrs))(*ptrs)

        entry_args: Tuple[Any, ...] = (
            inputs_arr,
            output.ctypes.data_as(ctypes.c_void_p),
            var_axes_arr,
        )
        return LlamaCppContext(
            entry_args=entry_args, output=output, _keepalive=keepalive
        )

    def unwrap_output(self, ctx: LlamaCppContext) -> np.ndarray:
        return ctx.output

    def release(self, ctx: LlamaCppContext) -> None:
        ctx._keepalive.clear()
