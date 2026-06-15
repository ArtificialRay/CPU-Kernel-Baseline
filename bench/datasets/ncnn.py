"""ncnn dataset adapter: numpy ↔ ncnn::Mat via ctypes.

The C-side implementation of these factory functions lives in
`bench/datasets/_ncnn_lib/_mat_factory.cpp`, which the NcnnBuilder
(`bench/compile/builders/ncnn.py`) compiles into every solution.so build. After
dlopening the solution.so, the runner resolves both:
  - armbench_entry_<op_type>  (from the solution's own binding.cpp)
  - armbench_ncnn_mat_*       (from _mat_factory.cpp)
…and ctypes-binds them. Every ncnn baseline ships a self-contained
armbench_entry_<op_type> that bakes its scalar params as constexpr, so the entry
takes only the 6 Mat/Option pointers — the runner declares that signature inline.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from bench.data.definition import Definition

# Slots the ncnn conv-family entry always expects even when absent from the
# definition (passed as empty 1D Mats). Ordered to match the harness ABI.
_NCNN_EXTRA_SLOTS = ("bias", "activation_params")

import numpy as np


# ── ncnn::Mat factory binding (per loaded .so) ───────────────────────────────

@dataclass
class _MatFactoryFns:
    """ctypes bindings to the armbench_ncnn_* symbols in a loaded .so."""

    create_3d: Any
    create_2d: Any
    create_1d: Any
    create_empty: Any
    destroy: Any
    read_3d: Any
    read_2d: Any
    read_1d: Any
    dims: Any
    w: Any
    h: Any
    c: Any
    empty: Any
    option_default: Any
    option_destroy: Any
    convert_packing: Any

    @classmethod
    def from_lib(cls, lib: ctypes.CDLL) -> "_MatFactoryFns":
        # Bind argtypes/restype for each function. Doing it once per .so keeps
        # the per-call paths clean.
        def _bind(name: str, restype, argtypes):
            f = getattr(lib, name)
            f.restype = restype
            f.argtypes = argtypes
            return f

        c_float_p = ctypes.POINTER(ctypes.c_float)
        return cls(
            create_3d=_bind("armbench_ncnn_mat_create_3d", ctypes.c_void_p,
                            [ctypes.c_int, ctypes.c_int, ctypes.c_int, c_float_p]),
            create_2d=_bind("armbench_ncnn_mat_create_2d", ctypes.c_void_p,
                            [ctypes.c_int, ctypes.c_int, c_float_p]),
            create_1d=_bind("armbench_ncnn_mat_create_1d", ctypes.c_void_p,
                            [ctypes.c_int, c_float_p]),
            create_empty=_bind("armbench_ncnn_mat_create_empty", ctypes.c_void_p, []),
            destroy=_bind("armbench_ncnn_mat_destroy", None, [ctypes.c_void_p]),
            read_3d=_bind("armbench_ncnn_mat_read_3d", ctypes.c_int,
                          [ctypes.c_void_p, c_float_p]),
            read_2d=_bind("armbench_ncnn_mat_read_2d", ctypes.c_int,
                          [ctypes.c_void_p, c_float_p]),
            read_1d=_bind("armbench_ncnn_mat_read_1d", ctypes.c_int,
                          [ctypes.c_void_p, c_float_p]),
            dims=_bind("armbench_ncnn_mat_dims", ctypes.c_int, [ctypes.c_void_p]),
            w=_bind("armbench_ncnn_mat_w", ctypes.c_int, [ctypes.c_void_p]),
            h=_bind("armbench_ncnn_mat_h", ctypes.c_int, [ctypes.c_void_p]),
            c=_bind("armbench_ncnn_mat_c", ctypes.c_int, [ctypes.c_void_p]),
            empty=_bind("armbench_ncnn_mat_empty", ctypes.c_int, [ctypes.c_void_p]),
            option_default=_bind("armbench_ncnn_option_create_default",
                                 ctypes.c_void_p, []),
            option_destroy=_bind("armbench_ncnn_option_destroy", None,
                                 [ctypes.c_void_p]),
            convert_packing=_bind("armbench_ncnn_convert_packing", ctypes.c_void_p,
                                  [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]),
        )


# ── Per-tensor packing logic ─────────────────────────────────────────────────

def _np_to_mat(fns: _MatFactoryFns, name: str, arr: np.ndarray) -> ctypes.c_void_p:
    """Construct an ncnn::Mat from a numpy array, choose 1D/2D/3D by tensor name+shape.

    Naming convention used by inputs.py:
      - 'input' / 'bottom_blob'  → 3D (collapse leading N into c if N=1)
      - 'weight' / 'weight_data' → flat 1D
      - 'bias' / 'bias_data'     → flat 1D
      - 'activation_params'      → flat 1D (or empty Mat if size 0)
    """
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    flat_ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    if name in ("input", "bottom_blob"):
        if arr.ndim == 4:
            # (N, C, H, W) → require N=1 for now; multi-batch needs Phase 3
            n, c, h, w = arr.shape
            if n != 1:
                raise NotImplementedError(
                    f"Phase 1 ncnn binding supports N=1 only; got input shape {arr.shape}"
                )
            return ctypes.c_void_p(fns.create_3d(w, h, c, flat_ptr))
        elif arr.ndim == 3:
            c, h, w = arr.shape
            return ctypes.c_void_p(fns.create_3d(w, h, c, flat_ptr))
        elif arr.ndim == 2:
            h, w = arr.shape
            return ctypes.c_void_p(fns.create_2d(w, h, flat_ptr))
        else:
            raise ValueError(f"Input '{name}' has unsupported rank {arr.ndim}")
    else:
        # weight / bias / activation_params → flat 1D
        return ctypes.c_void_p(fns.create_1d(int(arr.size), flat_ptr))


def _mat_to_np(fns: _MatFactoryFns, mat_ptr: ctypes.c_void_p) -> np.ndarray:
    """Read an ncnn::Mat back into a numpy array, choosing rank from Mat.dims."""
    dims = fns.dims(mat_ptr)
    w = fns.w(mat_ptr)
    h = fns.h(mat_ptr)
    c = fns.c(mat_ptr)
    if dims == 3:
        out = np.empty((c, h, w), dtype=np.float32)
        rc = fns.read_3d(mat_ptr, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    elif dims == 2:
        out = np.empty((h, w), dtype=np.float32)
        rc = fns.read_2d(mat_ptr, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    elif dims == 1:
        out = np.empty((w,), dtype=np.float32)
        rc = fns.read_1d(mat_ptr, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    else:
        raise ValueError(f"Unexpected Mat dims: {dims}")
    if rc != 0:
        raise RuntimeError(f"armbench_ncnn_mat_read_{dims}d returned {rc}")
    return out


# ── Dataset adapter ──────────────────────────────────────────────────────────

@dataclass
class NcnnContext:
    """Resources held between input wrap and output unwrap. Mat pointers must
    outlive the kernel call; we keep them here so the caller can release after.
    """

    fns: _MatFactoryFns
    mat_ptrs: List[ctypes.c_void_p]  # all created Mats (inputs + outputs + opt)
    output_mat_ptr: ctypes.c_void_p  # the empty Mat that the harness will .create() into
    opt_ptr: ctypes.c_void_p
    # Bound positional args ready to splat into the entry_point
    entry_args: Tuple[Any, ...]


class NcnnDataset:
    """Adapter for the ncnn dataset.

    Usage:
        ds = NcnnDataset()
        ctx = ds.wrap_inputs(definition, workload, np_inputs, op_type, lib)
        ret = entry(*ctx.entry_args)
        out = ds.unwrap_output(ctx)
        ds.release(ctx)
    """

    name = "ncnn"

    def __init__(self) -> None:
        pass

    def bind_lib(self, lib: ctypes.CDLL) -> _MatFactoryFns:
        """Bind the armbench_ncnn_* symbols in a freshly-dlopened solution.so."""
        return _MatFactoryFns.from_lib(lib)

    def wrap_inputs(
        self,
        np_inputs: Dict[str, np.ndarray],
        op_type: str,
        lib: ctypes.CDLL,
        *,
        definition: Definition,
        out_shape: Optional[Tuple[int, ...]] = None,
    ) -> NcnnContext:
        """Build the ctypes argument list for `armbench_entry_<op_type>`.

        Every ncnn baseline ships a self-contained entry that bakes its scalar
        params as constexpr, so the call passes only the Mat/Option pointers.
        Tensor order: definition.inputs (non-scalars) + any _NCNN_EXTRA_SLOTS
        not already present. out_shape is unused (ncnn allocates internally).
        """
        fns = self.bind_lib(lib)

        # Non-scalar inputs from definition, then append any extra slots
        # expected by the ncnn conv ABI that aren't in the definition.
        tensor_names: List[str] = [
            n for n, s in definition.inputs.items() if s.shape is not None
        ]
        for slot in _NCNN_EXTRA_SLOTS:
            if slot not in tensor_names:
                tensor_names.append(slot)

        created: List[ctypes.c_void_p] = []

        # Pack input tensors (bottom, weight, bias, act_params)
        tensor_ptrs: List[ctypes.c_void_p] = []
        for tname in tensor_names:
            arr = np_inputs.get(tname)
            if arr is None:
                # Allow missing tensors (e.g. no-bias case) → empty 1D Mat
                # NB: ncnn treats an empty Mat as "no bias" via bias_data.empty()
                ptr = ctypes.c_void_p(fns.create_1d(0, ctypes.POINTER(ctypes.c_float)()))
            else:
                ptr = _np_to_mat(fns, tname, arr)
            tensor_ptrs.append(ptr)
            created.append(ptr)

        # Output Mat (empty; harness will .create() it to correct shape)
        output_ptr = ctypes.c_void_p(fns.create_empty())
        created.append(output_ptr)

        # Default Option (use_packing_layout=true — see _mat_factory.cpp)
        opt_ptr = ctypes.c_void_p(fns.option_default())

        # Pack the input tensor to elempack=4 so ncnn's forward() takes the
        # optimised packed path.  Only applies to 3D Mats (conv2d/deconv2d
        # family); conv1d uses a 2D Mat whose packing semantics differ.
        # convert_packing returns a new heap Mat; add it to `created` so
        # release() destroys it.
        if fns.dims(tensor_ptrs[0]) == 3:
            packed_input_ptr = ctypes.c_void_p(
                fns.convert_packing(tensor_ptrs[0], 4, opt_ptr)
            )
            created.append(packed_input_ptr)
        else:
            packed_input_ptr = tensor_ptrs[0]

        # Assemble entry args. Order must match the C signature of
        # armbench_entry_<op_type> (6 Mat/Option pointers).
        base = (
            packed_input_ptr,  # bottom (pack4)
            output_ptr,        # top (output, empty — harness allocates)
            tensor_ptrs[1],    # weight
            tensor_ptrs[2],    # bias
            tensor_ptrs[3],    # activation_params
            opt_ptr,
        )
        # Self-contained entry: scalars are baked constexpr in the solution's
        # binding, so we pass only the 6 Mat/Option pointers.
        entry_args = base

        return NcnnContext(
            fns=fns,
            mat_ptrs=created,
            output_mat_ptr=output_ptr,
            opt_ptr=opt_ptr,
            entry_args=entry_args,
        )

    def unwrap_output(self, ctx: NcnnContext) -> np.ndarray:
        """Read the output Mat back into numpy, unpacking from pack4 to pack1 first."""
        if ctx.fns.empty(ctx.output_mat_ptr):
            raise RuntimeError("Output Mat is empty — kernel didn't allocate / returned non-zero")
        # Unpack only 3D output Mats; 2D output (conv1d) is already pack1.
        if ctx.fns.dims(ctx.output_mat_ptr) == 3:
            unpacked_ptr = ctypes.c_void_p(
                ctx.fns.convert_packing(ctx.output_mat_ptr, 1, ctx.opt_ptr)
            )
            try:
                return _mat_to_np(ctx.fns, unpacked_ptr)
            finally:
                ctx.fns.destroy(unpacked_ptr)
        return _mat_to_np(ctx.fns, ctx.output_mat_ptr)

    def release(self, ctx: NcnnContext) -> None:
        """Free all Mat / Option pointers we created during wrap_inputs."""
        for p in ctx.mat_ptrs:
            ctx.fns.destroy(p)
        ctx.fns.option_destroy(ctx.opt_ptr)
        ctx.mat_ptrs.clear()
