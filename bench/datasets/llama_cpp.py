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
  carries NULL. Q4_K tensor triplets (`<name>_q4`/`<name>_scales`/
  `<name>_mins`, a nibble-packed uint8 tensor immediately followed by two
  float16 tensors of the same leading shape) are repacked into ggml
  block_q4_K layout — 144-byte blocks along the last axis — and passed at
  the `_q4` tensor's slot; the two consumed scales/mins slots carry NULL.
  Const axes are baked as constexpr in the binding, so only buffers cross
  the ABI.
- `output`  — contiguous float32 buffer, allocated here from the reference
  output shape (ggml's [ne0..] reversed order equals numpy row-major).
- `var_axes` — values of the Definition's var axes, in declaration order
  (e.g. [M] for gemm, [M, S] for mha, [n_tokens] for moe).
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import ml_dtypes
import numpy as np

from bench.data.definition import Definition, DType

_Q8_0_BLOCK = 32
_Q8_0_BLOCK_BYTES = 34  # sizeof(block_q8_0) = 2 (fp16 d) + 32 (int8 qs)

_Q4_K_SUPERBLOCK = 256  # QK_K
_Q4_K_SUBBLOCK = 32
_Q4_K_BLOCK_BYTES = 144  # sizeof(block_q4_K) = 2+2 (fp16 d,dmin) + 12 (scales) + 128 (qs)

# Plain (non-q8_0-paired) tensor dtypes this adapter can pack, mapped to the
# numpy dtype used to make the buffer contiguous before taking its pointer.
_PLAIN_DTYPE_TO_NP: Dict[DType, Any] = {
    DType.FLOAT32: np.float32,
    DType.BFLOAT16: ml_dtypes.bfloat16,
}


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


def _repack_q4_k(q4: np.ndarray, scales: np.ndarray, mins: np.ndarray) -> np.ndarray:
    """Repack (flat nibbles, flat per-32-elem scale, flat per-32-elem min)
    into real ggml block_q4_K rows.

    Definition ABI (flat/simple, see llama_cpp.py module docstring):
      q4     [..., K/2]  uint8    — byte i holds element 2i (low nibble),
                                     2i+1 (high nibble), plain sequential order.
      scales [..., K/32] float16 — per-32-elem sub-block scale, already
                                     combining ggml's d*sc into one number.
      mins   [..., K/32] float16 — per-32-elem sub-block min, already
                                     combining ggml's dmin*m.

    Output: uint8 [..., K/256, 144], each 144-byte block_q4_K:
      {fp16 d; fp16 dmin; uint8 scales[12] (6-bit packed); uint8 qs[128]}
    Re-derives d/dmin/6-bit sc/m per super-block from the flat scale/min
    values (max-scale/63 like real `quantize_row_q4_K_ref`) — this is the
    same lossy 6-bit compression real Q4_K quantization performs, not an
    approximation introduced here.
    """
    *lead, half = q4.shape
    k = half * 2
    if k % _Q4_K_SUPERBLOCK != 0:
        raise ValueError(f"q4_k tensor last dim {k} not a multiple of {_Q4_K_SUPERBLOCK}")
    nb = k // _Q4_K_SUPERBLOCK
    nsub = k // _Q4_K_SUBBLOCK
    if tuple(scales.shape) != (*lead, nsub) or tuple(mins.shape) != (*lead, nsub):
        raise ValueError(
            f"q4_k scales/mins shape {scales.shape}/{mins.shape} does not match "
            f"quants shape {q4.shape} (expected {(*lead, nsub)})"
        )
    rows = int(np.prod(lead)) if lead else 1

    q4r = np.ascontiguousarray(q4, dtype=np.uint8).reshape(rows, half)
    sc_flat = np.ascontiguousarray(scales, dtype=np.float32).reshape(rows, nb, 8)
    mn_flat = np.ascontiguousarray(mins, dtype=np.float32).reshape(rows, nb, 8)

    # Unpack flat sequential nibbles -> per-element values [rows, K].
    lo = q4r & 0x0F
    hi = (q4r >> 4) & 0x0F
    elems = np.empty((rows, k), dtype=np.uint8)
    elems[:, 0::2] = lo
    elems[:, 1::2] = hi
    elems = elems.reshape(rows, nb, _Q4_K_SUPERBLOCK)

    # Re-derive per-super-block d/dmin (fp16) and per-sub-block 6-bit sc/m,
    # mirroring quantize_row_q4_K_ref's encode step exactly.
    max_scale = sc_flat.max(axis=-1)  # [rows, nb]
    max_min = mn_flat.max(axis=-1)
    d = (max_scale / 63.0).astype(np.float32)
    dmin = (max_min / 63.0).astype(np.float32)
    inv_d = np.where(max_scale > 0, 63.0 / np.maximum(max_scale, 1e-30), 0.0)
    inv_dmin = np.where(max_min > 0, 63.0 / np.maximum(max_min, 1e-30), 0.0)
    ls = np.clip(np.round(inv_d[..., None] * sc_flat), 0, 63).astype(np.uint8)  # [rows,nb,8]
    lm = np.clip(np.round(inv_dmin[..., None] * mn_flat), 0, 63).astype(np.uint8)

    packed_scales = np.zeros((rows, nb, 12), dtype=np.uint8)
    packed_scales[:, :, 0:4] = ls[:, :, 0:4]
    packed_scales[:, :, 4:8] = lm[:, :, 0:4]
    packed_scales[:, :, 8:12] = (ls[:, :, 4:8] & 0xF) | ((lm[:, :, 4:8] & 0xF) << 4)
    packed_scales[:, :, 0:4] |= (ls[:, :, 4:8] >> 4) << 6
    packed_scales[:, :, 4:8] |= (lm[:, :, 4:8] >> 4) << 6

    # Nibble-pack qs in ggml's low-half/high-half order: within each 64-elem
    # chunk, byte l holds element l (low nibble) and element l+32 (high nibble).
    qs = np.zeros((rows, nb, 128), dtype=np.uint8)
    for c in range(4):
        j = c * 64
        lo_half = elems[:, :, j : j + 32]
        hi_half = elems[:, :, j + 32 : j + 64]
        qs[:, :, c * 32 : (c + 1) * 32] = (lo_half & 0xF) | (hi_half << 4)

    d_bytes = d.astype("<f2").view(np.uint8).reshape(rows, nb, 2)
    dmin_bytes = dmin.astype("<f2").view(np.uint8).reshape(rows, nb, 2)

    buf = np.empty((rows, nb, _Q4_K_BLOCK_BYTES), dtype=np.uint8)
    buf[:, :, 0:2] = d_bytes
    buf[:, :, 2:4] = dmin_bytes
    buf[:, :, 4:16] = packed_scales
    buf[:, :, 16:144] = qs
    return buf.reshape(*lead, nb, _Q4_K_BLOCK_BYTES)


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
            nxt2 = tensor_specs[i + 2] if i + 2 < len(tensor_specs) else None
            if (
                spec.dtype == DType.UINT8
                and nxt is not None
                and nxt[1].dtype == DType.FLOAT16
                and nxt2 is not None
                and nxt2[1].dtype == DType.FLOAT16
            ):
                # Q4_K triplet: nibbles + scales + mins → one repacked
                # block_q4_K buffer at the nibbles slot, NULL at the two
                # consumed scales/mins slots.
                scales = np_inputs.get(nxt[0])
                mins = np_inputs.get(nxt2[0])
                if scales is None or mins is None:
                    raise ValueError(
                        f"Missing Q4_K scales/mins tensor '{nxt[0]}'/'{nxt2[0]}' "
                        f"for '{definition.name}'"
                    )
                packed = _repack_q4_k(np.asarray(arr), np.asarray(scales), np.asarray(mins))
                keepalive.append(packed)
                ptrs.append(packed.ctypes.data)
                ptrs.append(None)
                ptrs.append(None)
                i += 3
                continue
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
            np_dtype = _PLAIN_DTYPE_TO_NP.get(spec.dtype)
            if np_dtype is None:
                raise NotImplementedError(
                    f"LlamaCppDataset: tensor '{name}' has dtype {spec.dtype!r}, which "
                    f"isn't a plain (non-q8_0-paired) dtype this adapter supports "
                    f"(supported: {[d.value for d in _PLAIN_DTYPE_TO_NP]})"
                )
            contiguous = np.ascontiguousarray(arr, dtype=np_dtype)
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
