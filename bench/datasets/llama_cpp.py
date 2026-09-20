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
  Q5_K triplets (`<name>_q5`/`<name>_scales`/`<name>_mins`: a uint8 tensor
  with one 5-bit value per element [..., K] followed by two float16 tensors
  [..., K/32] -- told apart from Q4_K by the quant tensor being 32x, not 16x,
  the scales' last dim) are repacked into ggml block_q5_K layout -- 176-byte
  blocks along the last axis -- with the same NULL convention. Q6_K triplets
  (`<name>_q6`/`<name>_scales`/`<name>_d`: a uint8 tensor with one 6-bit
  value per element [..., K], an int8 per-16-elem scales tensor [..., K/16]
  and a float16 per-256-elem super-block scale tensor [..., K/256]) are
  repacked into ggml block_q6_K layout -- 210-byte blocks along the last
  axis -- likewise.
  A uint8 tensor that is NOT the head of such a triplet (the following
  tensors' dtypes decide) is passed through as a contiguous uint8 buffer:
  layout=ggml definitions (`gemm_ggml_q4_K_*`, `gemm_ggml_q5_K_*`,
  `gemm_ggml_q6_K_*`, whose `B` is uint8 [N, K/256 * sizeof(block)]) pass
  block rows as-is -- no repack, no NULL slots.
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

_Q5_K_SUPERBLOCK = 256  # QK_K
_Q5_K_SUBBLOCK = 32
_Q5_K_BLOCK_BYTES = 176  # sizeof(block_q5_K) = 2+2 (fp16 d,dmin) + 12 (scales) + 32 (qh) + 128 (qs)

_Q6_K_SUPERBLOCK = 256  # QK_K
_Q6_K_SUBBLOCK = 16
_Q6_K_BLOCK_BYTES = 210  # sizeof(block_q6_K) = 128 (ql) + 64 (qh) + 16 (int8 scales) + 2 (fp16 d)

# Plain (non-q8_0-paired) tensor dtypes this adapter can pack, mapped to the
# numpy dtype used to make the buffer contiguous before taking its pointer.
_PLAIN_DTYPE_TO_NP: Dict[DType, Any] = {
    DType.FLOAT32: np.float32,
    DType.BFLOAT16: ml_dtypes.bfloat16,
    DType.UINT8: np.uint8,  # e.g. pre-packed ggml block rows (layout=ggml definitions)
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


def _kquant_encode_scales_mins(
    sc_flat: np.ndarray, mn_flat: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shared Q4_K/Q5_K super-block scale/min encode.

    sc_flat/mn_flat: float32 [rows, nb, 8] -- the flat per-32-elem scale and
    min of every sub-block (already d*sc and dmin*m in ggml terms).
    Returns (d [rows,nb] f32, dmin [rows,nb] f32, ls [rows,nb,8] u8,
    lm [rows,nb,8] u8, packed_scales [rows,nb,12] u8): the per-super-block
    fp16-bound d/dmin (max/63, like quantize_row_q4_K_ref /
    quantize_row_q5_K_ref), the 6-bit sub-block factors, and the 12-byte
    `scales` field in ggml's get_scale_min_k4 packing (sub-blocks 0-3 take
    the low 6 bits of bytes 0-3 (sc) / 4-7 (m); sub-blocks 4-7 take the
    nibbles of bytes 8-11 plus the top 2 bits of bytes 0-3 / 4-7).
    """
    rows, nb, _ = sc_flat.shape
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
    return d, dmin, ls, lm, packed_scales


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
    d, dmin, _ls, _lm, packed_scales = _kquant_encode_scales_mins(sc_flat, mn_flat)

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


def _repack_q5_k(q5: np.ndarray, scales: np.ndarray, mins: np.ndarray) -> np.ndarray:
    """Repack (flat 5-bit values, flat per-32-elem scale, flat per-32-elem min)
    into real ggml block_q5_K rows.

    Definition ABI (flat/simple, see llama_cpp.py module docstring):
      q5     [..., K]    uint8    — one value per element (0..31; only the low
                                     5 bits of each byte are the quant, higher
                                     bits are masked off), sequential order.
      scales [..., K/32] float16 — per-32-elem sub-block scale (ggml's d*sc).
      mins   [..., K/32] float16 — per-32-elem sub-block min (ggml's dmin*m).

    Output: uint8 [..., K/256, 176], each 176-byte block_q5_K:
      {fp16 d; fp16 dmin; uint8 scales[12] (6-bit packed); uint8 qh[32];
       uint8 qs[128]}
    d/dmin and the 6-bit sc/m are re-derived per super-block exactly like
    `_repack_q4_k` (same lossy compression real Q5_K quantization performs).
    Bit placement follows ggml's dequantize_row_q5_K: within each 64-element
    chunk c (4 per super-block), byte qs[c*32+l] holds the low nibble of
    element c*64+l (low half) and of element c*64+32+l (high half), and the
    5th bit of those two elements lives in qh[l] at bit 2c and bit 2c+1.
    """
    *lead, k = q5.shape
    if k % _Q5_K_SUPERBLOCK != 0:
        raise ValueError(f"q5_k tensor last dim {k} not a multiple of {_Q5_K_SUPERBLOCK}")
    nb = k // _Q5_K_SUPERBLOCK
    nsub = k // _Q5_K_SUBBLOCK
    if tuple(scales.shape) != (*lead, nsub) or tuple(mins.shape) != (*lead, nsub):
        raise ValueError(
            f"q5_k scales/mins shape {scales.shape}/{mins.shape} does not match "
            f"quants shape {q5.shape} (expected {(*lead, nsub)})"
        )
    rows = int(np.prod(lead)) if lead else 1

    elems = (np.ascontiguousarray(q5, dtype=np.uint8) & 0x1F).reshape(rows, nb, _Q5_K_SUPERBLOCK)
    sc_flat = np.ascontiguousarray(scales, dtype=np.float32).reshape(rows, nb, 8)
    mn_flat = np.ascontiguousarray(mins, dtype=np.float32).reshape(rows, nb, 8)

    d, dmin, _ls, _lm, packed_scales = _kquant_encode_scales_mins(sc_flat, mn_flat)

    qs = np.zeros((rows, nb, 128), dtype=np.uint8)
    qh = np.zeros((rows, nb, 32), dtype=np.uint8)
    for c in range(4):
        j = c * 64
        lo_half = elems[:, :, j : j + 32]
        hi_half = elems[:, :, j + 32 : j + 64]
        qs[:, :, c * 32 : (c + 1) * 32] = (lo_half & 0xF) | ((hi_half & 0xF) << 4)
        qh |= ((lo_half >> 4) & 1) << (2 * c)
        qh |= ((hi_half >> 4) & 1) << (2 * c + 1)

    d_bytes = d.astype("<f2").view(np.uint8).reshape(rows, nb, 2)
    dmin_bytes = dmin.astype("<f2").view(np.uint8).reshape(rows, nb, 2)

    buf = np.empty((rows, nb, _Q5_K_BLOCK_BYTES), dtype=np.uint8)
    buf[:, :, 0:2] = d_bytes
    buf[:, :, 2:4] = dmin_bytes
    buf[:, :, 4:16] = packed_scales
    buf[:, :, 16:48] = qh
    buf[:, :, 48:176] = qs
    return buf.reshape(*lead, nb, _Q5_K_BLOCK_BYTES)


def _repack_q6_k(q6: np.ndarray, scales: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Repack (flat 6-bit values, flat per-16-elem int8 scale, per-256-elem
    fp16 super-block scale) into real ggml block_q6_K rows.

    Definition ABI (flat/simple, see llama_cpp.py module docstring):
      q6     [..., K]     uint8   — one value per element (0..63; only the low
                                      6 bits of each byte are the quant, higher
                                      bits are masked off), sequential order.
                                      Dequant is d*sc*(q - 32) like ggml.
      scales [..., K/16]  int8    — per-16-elem sub-block scale (ggml's
                                      `scales`, used verbatim).
      d      [..., K/256] float16 — per-super-block scale (ggml's `d`, used
                                      verbatim).

    Output: uint8 [..., K/256, 210], each 210-byte block_q6_K:
      {uint8 ql[128]; uint8 qh[64]; int8 scales[16]; fp16 d}
    This repack is lossless (no re-derivation: Q6_K stores its sub-block
    scales as plain int8). Bit placement follows ggml's dequantize_row_q6_K:
    each super-block is two 128-element halves h; within a half, for l in
    0..31, elements h*128 + {l, 32+l, 64+l, 96+l} put their low nibbles in
    ql[h*64+l] (low), ql[h*64+32+l] (low), ql[h*64+l] (high),
    ql[h*64+32+l] (high) and their top 2 bits in qh[h*32+l] at bit pairs
    0-1, 2-3, 4-5, 6-7 respectively. scales[16] is plain sequential
    (sub-block j of the super-block).
    """
    *lead, k = q6.shape
    if k % _Q6_K_SUPERBLOCK != 0:
        raise ValueError(f"q6_k tensor last dim {k} not a multiple of {_Q6_K_SUPERBLOCK}")
    nb = k // _Q6_K_SUPERBLOCK
    nsub = k // _Q6_K_SUBBLOCK
    if tuple(scales.shape) != (*lead, nsub):
        raise ValueError(
            f"q6_k scales shape {scales.shape} does not match quants shape "
            f"{q6.shape} (expected {(*lead, nsub)})"
        )
    if tuple(d.shape) != (*lead, nb):
        raise ValueError(
            f"q6_k d shape {d.shape} does not match quants shape "
            f"{q6.shape} (expected {(*lead, nb)})"
        )
    rows = int(np.prod(lead)) if lead else 1

    elems = (np.ascontiguousarray(q6, dtype=np.uint8) & 0x3F).reshape(rows, nb, _Q6_K_SUPERBLOCK)
    sc = np.ascontiguousarray(scales, dtype=np.int8).reshape(rows, nb, 16)
    dd = np.ascontiguousarray(d, dtype=np.dtype("<f2")).reshape(rows, nb)

    ql = np.zeros((rows, nb, 128), dtype=np.uint8)
    qh = np.zeros((rows, nb, 64), dtype=np.uint8)
    for h in range(2):
        e = elems[:, :, h * 128 : (h + 1) * 128]
        q1 = e[:, :, 0:32]
        q2 = e[:, :, 32:64]
        q3 = e[:, :, 64:96]
        q4 = e[:, :, 96:128]
        ql[:, :, h * 64 : h * 64 + 32] = (q1 & 0xF) | ((q3 & 0xF) << 4)
        ql[:, :, h * 64 + 32 : h * 64 + 64] = (q2 & 0xF) | ((q4 & 0xF) << 4)
        qh[:, :, h * 32 : (h + 1) * 32] = (
            (q1 >> 4) | ((q2 >> 4) << 2) | ((q3 >> 4) << 4) | ((q4 >> 4) << 6)
        )

    buf = np.empty((rows, nb, _Q6_K_BLOCK_BYTES), dtype=np.uint8)
    buf[:, :, 0:128] = ql
    buf[:, :, 128:192] = qh
    buf[:, :, 192:208] = sc.view(np.uint8)
    buf[:, :, 208:210] = dd.view(np.uint8).reshape(rows, nb, 2)
    return buf.reshape(*lead, nb, _Q6_K_BLOCK_BYTES)


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
                and nxt2 is not None
                and nxt2[1].dtype == DType.FLOAT16
                and nxt[1].dtype in (DType.FLOAT16, DType.INT8)
            ):
                # k-quant triplet: quants + two side tensors → one repacked
                # block_qX_K buffer at the quants slot, NULL at the two
                # consumed slots.
                #   UINT8 + FLOAT16 + FLOAT16, quants 16x scales' last dim →
                #     Q4_K (`_q4` nibbles + `_scales` + `_mins`)
                #   UINT8 + FLOAT16 + FLOAT16, quants 32x scales' last dim →
                #     Q5_K (`_q5` bytes + `_scales` + `_mins`)
                #   UINT8 + INT8 + FLOAT16 →
                #     Q6_K (`_q6` bytes + int8 `_scales` + fp16 `_d`)
                side1 = np_inputs.get(nxt[0])
                side2 = np_inputs.get(nxt2[0])
                if side1 is None or side2 is None:
                    raise ValueError(
                        f"Missing k-quant side tensor '{nxt[0]}'/'{nxt2[0]}' "
                        f"for '{definition.name}'"
                    )
                q = np.asarray(arr)
                side1 = np.asarray(side1)
                side2 = np.asarray(side2)
                if nxt[1].dtype == DType.INT8:
                    packed = _repack_q6_k(q, side1, side2)
                elif q.shape[-1] == 16 * side1.shape[-1]:
                    packed = _repack_q4_k(q, side1, side2)
                elif q.shape[-1] == 32 * side1.shape[-1]:
                    packed = _repack_q5_k(q, side1, side2)
                else:
                    raise ValueError(
                        f"k-quant tensor '{name}' last dim {q.shape[-1]} is neither "
                        f"16x (Q4_K) nor 32x (Q5_K) the scales' last dim "
                        f"{side1.shape[-1]} for '{definition.name}'"
                    )
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
