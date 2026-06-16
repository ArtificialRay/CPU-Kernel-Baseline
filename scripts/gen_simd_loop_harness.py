#!/usr/bin/env python3
"""Generate simd-loop harness + bench-trace artifacts for all supported loops.

Run from repo root:
    python scripts/gen_simd_loop_harness.py

Outputs (all idempotent — safe to re-run):
  bench/compile/builders/simd_loop_harness/loop_NNN.{h,cpp}
  bench-trace/definitions/simd-loop/loop_NNN.json
  bench-trace/workloads/simd-loop/loop_NNN.jsonl
  bench-trace/solutions/simd-loop/reference-scalar/loop_NNN/reference-scalar_loop_NNN.json
  bench/datasets/simd_loop.py   ← fully regenerated

Supported loop patterns (auto-generated):
  A  — two pointer arrays  + int n + scalar output
  B  — one pointer array   + int n + scalar output
  C  — N input arrays + one output array + int n  (array-output, last ptr = output)
  D  — one input array + one output array + int n  (array-output)

Loops requiring custom handling (skipped):
  loop_023 — indexes OOB with generated inputs
  loop_101 — output array is FIRST ptr, size ≠ N
  loop_105 — b is scratch buffer, not an input
  loop_111 — last ptr is input (exponent), not output
  p+lmt loops (005, 006, 022, 034, 103) — need string/buffer input generation
  matrix multiply (025, 130, 135-137, 201-221, 223, 231, 245) — 2D m/n/k axes
  complex structs (012, 019, 037, 109, 110, 112, 204, 211, 222) — custom C types
"""
from __future__ import annotations

import json
import re
import uuid as _uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

REPO          = Path(__file__).resolve().parent.parent
PROBLEMS_DIR  = REPO / "dataset" / "problems"
HARNESS_DIR   = REPO / "bench" / "compile" / "builders" / "simd_loop_harness"
BENCH_TRACE   = REPO / "bench-trace"
SIMD_LOOP_PY  = REPO / "bench" / "datasets" / "simd_loop.py"
LOOPS_DIR     = REPO / "loops"

# ── C type → numpy dtype string ──────────────────────────────────────────────

_C_TO_NUMPY = {
    "float":    "float32",
    "double":   "float64",
    "int8_t":   "int8",   "int16_t": "int16",
    "int32_t":  "int32",  "int64_t": "int64",
    "uint8_t":  "int8",   # stored as int8 (bit-compatible for values 1-100)
    "uint16_t": "int16",
    "uint32_t": "int32",  # stored as int32 (bit-compatible for values 1-100)
    "uint64_t": "int64",
    "int":      "int32",
    "bool":     "bool",
}

_C_TO_NP_RESULT = {
    "float":    "np.float32",
    "double":   "np.float64",
    "int8_t":   "np.int8",   "int16_t": "np.int16",
    "int32_t":  "np.int32",  "int64_t": "np.int64",
    "uint8_t":  "np.uint8",
    "uint16_t": "np.uint16",
    "uint32_t": "np.uint32",
    "uint64_t": "np.uint64",
    "int":      "np.int32",
    "bool":     "np.bool_",
}

_C_ZERO = {
    "float": "0.0f", "double": "0.0",
    "int": "0", "uint32_t": "0u", "uint64_t": "0ull",
    "int32_t": "0", "int64_t": "0LL",
    "uint8_t": "0", "uint16_t": "0",
    "bool": "0",
}

# ── Struct field ──────────────────────────────────────────────────────────────

@dataclass
class Field:
    c_type: str       # e.g. "float", "uint32_t"
    name:   str       # e.g. "a", "n", "res"
    is_ptr: bool      # True if pointer field

    @property
    def numpy_dtype(self) -> str:
        return _C_TO_NUMPY.get(self.c_type, "int32")

    @property
    def np_result_expr(self) -> str:
        return _C_TO_NP_RESULT.get(self.c_type, "np.int32")

    @property
    def c_zero(self) -> str:
        return _C_ZERO.get(self.c_type, "0")


@dataclass
class LoopInfo:
    loop_id:        str               # "loop_001"
    loop_num:       str               # "001"
    ptr_fields:     List[Field]       # ALL pointer fields in struct order
    n_field:        Field             # the integer size field
    res_field:      Optional[Field]   # scalar result field (None for array/inplace)
    output_ptr:     Optional[Field]   # array-output: struct field for the output array
    scratch_fields: List[Field]       # inplace: scratch ptrs (allocated, not input/output)
    edge_sizes:     List[int]         # from EDGE_SIZES (0 excluded)
    perf_sizes:     List[int]         # from PERF_SIZES
    kind:           str = "scalar"    # "scalar" | "array" | "inplace"

    @property
    def is_array_output(self) -> bool:
        return self.kind == "array"

    @property
    def is_inplace(self) -> bool:
        return self.kind == "inplace"

    @property
    def input_ptr_fields(self) -> List[Field]:
        """Pointer fields that are meaningful INPUTS (excludes output/scratch)."""
        if self.kind == "array":
            return [f for f in self.ptr_fields if f.name != self.output_ptr.name]
        if self.kind == "inplace":
            scratch_names = {f.name for f in self.scratch_fields}
            return [f for f in self.ptr_fields if f.name not in scratch_names]
        return self.ptr_fields


# ── Parser ────────────────────────────────────────────────────────────────────

def _parse_struct(struct_def: str) -> List[Field]:
    fields = []
    for line in struct_def.splitlines():
        line = line.strip().rstrip(";")
        if not line or "{" in line or "}" in line:
            continue
        # Remove restrict / __restrict__
        line = re.sub(r'\brestrict\b|__restrict__', '', line).strip()
        # Pointer: "type *name" or "type* name"
        m = re.match(r'([\w]+)\s*\*+\s*([\w]+)$', line)
        if m:
            fields.append(Field(c_type=m.group(1), name=m.group(2), is_ptr=True))
            continue
        # Scalar: "type name"
        m = re.match(r'([\w]+)\s+([\w]+)$', line)
        if m:
            fields.append(Field(c_type=m.group(1), name=m.group(2), is_ptr=False))
    return fields


_SIZE_NAMES = {"n", "size", "len", "length", "count", "num"}
_RES_NAMES  = {"res", "result", "checksum", "sum", "out", "output"}

# C type → numpy dtype for array outputs (uses proper unsigned types unlike scalar map)
_array_dtype_map_local = {
    "float": "float32", "double": "float64",
    "int": "int32", "int32_t": "int32", "int64_t": "int64",
    "uint32_t": "uint32", "uint64_t": "uint64",
    "uint8_t": "uint8", "uint16_t": "uint16",
    "int8_t": "int8", "int16_t": "int16",
    "bool": "int32",
}

# In-place sort loops: only "data" is meaningful input/output; extra ptrs are scratch.
# Custom scalar kernel (std::sort) is injected instead of extracting from loops/*.c.
_SORT_LOOPS: dict[str, list] = {
    "loop_120": [],                       # no scratch buffers
    "loop_121": ["temp"],                 # temp is scratch
    "loop_122": [],
    "loop_123": ["temp", "block_sizes"],  # bitonic sort; edge sizes are all powers-of-2
    "loop_124": ["temp", "hist", "prfx"], # radix sort; hist/prfx oversized to n (SVE uses ≤mvl*16)
}


def _classify(loop_id: str, fields: List[Field]) -> Optional[LoopInfo]:
    """Classify struct fields into a LoopInfo. Returns None if unsupported."""
    loop_num = re.search(r'loop_(\d+)', loop_id).group(1)

    ptr_fields    = [f for f in fields if f.is_ptr]
    scalar_fields = [f for f in fields if not f.is_ptr]

    if not ptr_fields or not scalar_fields:
        return None

    # Load sizes early (needed for all patterns)
    import importlib.util
    prob_dir = next((d for d in PROBLEMS_DIR.iterdir()
                     if d.name.startswith(loop_id + "_")), None)
    if not prob_dir:
        return None
    spec = importlib.util.spec_from_file_location("prob", prob_dir / "problem.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    edge = [s for s in getattr(mod, "EDGE_SIZES", []) if s > 0]
    perf = getattr(mod, "PERF_SIZES_C8G", None) or getattr(mod, "PERF_SIZES", [])

    def _mk(**kw):
        defaults = dict(res_field=None, output_ptr=None, scratch_fields=[], kind="scalar")
        defaults.update(kw)
        return LoopInfo(loop_id=loop_id, loop_num=loop_num, ptr_fields=ptr_fields,
                        edge_sizes=edge, perf_sizes=perf, **defaults)

    # ── In-place sort pattern (data is both input and output) ─────────────────
    if loop_id in _SORT_LOOPS:
        scratch_names = set(_SORT_LOOPS[loop_id])
        scratch = [f for f in ptr_fields if f.name in scratch_names]
        n_f = next((f for f in scalar_fields if f.name.lower() in _SIZE_NAMES), None)
        if n_f is None and scalar_fields:
            n_f = scalar_fields[0]
        if n_f is None:
            return None
        return _mk(n_field=n_f, scratch_fields=scratch, kind="inplace")

    # ── Generic patterns ──────────────────────────────────────────────────────
    res_field = None
    n_field   = None

    for f in scalar_fields:
        if f.name.lower() in _SIZE_NAMES or f.name in ("n", "size"):
            n_field = f
        elif f.name.lower() in _RES_NAMES:
            res_field = f

    if n_field is None:
        non_res = [f for f in scalar_fields if f != res_field]
        if non_res:
            n_field = non_res[0]

    if res_field is None:
        non_n = [f for f in scalar_fields if f != n_field]
        if non_n:
            res_field = non_n[-1]

    if n_field is None:
        return None

    # Scalar-output: has a res scalar + ≤2 input pointers
    if res_field is not None and len(ptr_fields) <= 2:
        return _mk(n_field=n_field, res_field=res_field, kind="scalar")

    # Array-output: no scalar res, 2+ ptrs, last ptr is the output.
    # ABI: armbench_entry(in1, ..., int64_t n, void* res_out) where res_out
    # is the pre-allocated N-element output buffer.
    if res_field is None and len(ptr_fields) >= 2:
        return _mk(n_field=n_field, output_ptr=ptr_fields[-1], kind="array")

    return None


# ── Harness generator ─────────────────────────────────────────────────────────

def _gen_harness_h(info: LoopInfo) -> str:
    lid = info.loop_id

    struct_fields = []
    if info.is_inplace:
        # n may appear first in struct for sort loops
        if info.ptr_fields and info.ptr_fields[0].name == info.n_field.name:
            struct_fields.append(f"    {info.n_field.c_type} {info.n_field.name};")
            for f in info.ptr_fields:
                struct_fields.append(f"    {f.c_type} *{f.name};")
        else:
            for f in info.ptr_fields:
                struct_fields.append(f"    {f.c_type} *{f.name};")
            struct_fields.append(f"    {info.n_field.c_type} {info.n_field.name};")
    else:
        for f in info.ptr_fields:
            struct_fields.append(f"    {f.c_type} *{f.name};")
        struct_fields.append(f"    {info.n_field.c_type} {info.n_field.name};")
        if info.kind == "scalar":
            struct_fields.append(f"    {info.res_field.c_type} {info.res_field.name};")

    if info.is_inplace:
        # ABI: data ptr + scratch ptrs + int64_t n + void* unused
        all_ptrs = info.input_ptr_fields + info.scratch_fields
        args = ", ".join(["void *" + f.name for f in all_ptrs] + ["int64_t n", "void *unused"])
    else:
        # ABI: input ptrs + int64_t n + void* res_out
        args = ", ".join(
            ["void *" + f.name for f in info.input_ptr_fields]
            + ["int64_t n", "void *res_out"]
        )

    return f"""\
// Auto-generated by scripts/gen_simd_loop_harness.py — do not hand-edit.
#pragma once
#include <stdint.h>

struct {lid}_data {{
{chr(10).join(struct_fields)}
}};

#ifdef __cplusplus
extern "C" {{
#endif
int armbench_entry_{lid}({args});
#ifdef __cplusplus
}}
#endif
"""


def _gen_harness_cpp(info: LoopInfo) -> str:
    lid = info.loop_id
    n_cast = f"static_cast<{info.n_field.c_type}>(n)"

    if info.is_inplace:
        all_ptrs = info.input_ptr_fields + info.scratch_fields
        args = ", ".join(["void *" + f.name for f in all_ptrs] + ["int64_t n", "void *unused"])
        # Use 'd' as local struct name to avoid shadowing pointer params like 'data'
        ptr_assigns = "\n    ".join(
            f"d.{f.name} = static_cast<{f.c_type} *>({f.name});"
            for f in info.ptr_fields
        )
        return f"""\
// Auto-generated by scripts/gen_simd_loop_harness.py — do not hand-edit.
#include "{lid}.h"
#include <stdint.h>

extern "C" void inner_{lid}(struct {lid}_data *data);

extern "C" int armbench_entry_{lid}({args}) {{
    struct {lid}_data d;
    d.{info.n_field.name} = {n_cast};
    {ptr_assigns}
    inner_{lid}(&d);
    return 0;
}}
"""

    args = ", ".join(
        ["void *" + f.name for f in info.input_ptr_fields]
        + ["int64_t n", "void *res_out"]
    )

    if info.is_array_output:
        # Map each input ptr and the output ptr (res_out → data.output_field)
        ptr_assigns = "\n    ".join(
            f"data.{f.name} = static_cast<{f.c_type} *>({f.name});"
            for f in info.input_ptr_fields
        )
        out = info.output_ptr
        return f"""\
// Auto-generated by scripts/gen_simd_loop_harness.py — do not hand-edit.
#include "{lid}.h"
#include <stdint.h>

extern "C" void inner_{lid}(struct {lid}_data *data);

extern "C" int armbench_entry_{lid}({args}) {{
    struct {lid}_data data;
    {ptr_assigns}
    data.{out.name} = static_cast<{out.c_type} *>(res_out);
    data.{info.n_field.name} = {n_cast};
    inner_{lid}(&data);
    return 0;
}}
"""
    else:
        ptr_assigns = "\n    ".join(
            f"data.{f.name} = static_cast<{f.c_type} *>({f.name});"
            for f in info.ptr_fields
        )
        res_type = info.res_field.c_type
        return f"""\
// Auto-generated by scripts/gen_simd_loop_harness.py — do not hand-edit.
#include "{lid}.h"
#include <cassert>
#include <limits>

extern "C" void inner_{lid}(struct {lid}_data *data);

extern "C" int armbench_entry_{lid}({args}) {{
    if (n < 0 || n > std::numeric_limits<int>::max()) return -1;
    struct {lid}_data data;
    {ptr_assigns}
    data.{info.n_field.name} = {n_cast};
    data.{info.res_field.name} = {info.res_field.c_zero};
    inner_{lid}(&data);
    *static_cast<{res_type} *>(res_out) = data.{info.res_field.name};
    return 0;
}}
"""


# ── Reference Python generator ────────────────────────────────────────────────

# Custom scalar kernels: used when _extract_scalar_kernel doesn't apply
# (e.g. sort loops that use complex helper chains from sort.h).
_CUSTOM_SCALAR_KERNELS: dict[str, str] = {
    # Python reference computes in float64 then casts to float32; match that
    # precision here so correctness comparison passes.
    "loop_001": (
        '#include "loop_001.h"\n'
        '#include <stdint.h>\n\n'
        'extern "C" void inner_loop_001(struct loop_001_data *data) {\n'
        '    float *a = data->a;\n'
        '    float *b = data->b;\n'
        '    int n = data->n;\n'
        '    double res = 0.0;\n'
        '    for (int i = 0; i < n; i++) {\n'
        '        res += (double)a[i] * (double)b[i];\n'
        '    }\n'
        '    data->res = (float)res;\n'
        '}\n'
    ),
    "loop_120": (
        '#include "loop_120.h"\n'
        '#include <algorithm>\n\n'
        'extern "C" void inner_loop_120(struct loop_120_data *data) {\n'
        '    std::sort(data->data, data->data + data->n);\n'
        '}\n'
    ),
    "loop_121": (
        '#include "loop_121.h"\n'
        '#include <algorithm>\n\n'
        'extern "C" void inner_loop_121(struct loop_121_data *data) {\n'
        '    std::sort(data->data, data->data + data->n);\n'
        '}\n'
    ),
    "loop_122": (
        '#include "loop_122.h"\n'
        '#include <algorithm>\n\n'
        'extern "C" void inner_loop_122(struct loop_122_data *data) {\n'
        '    std::sort(data->data, data->data + data->n);\n'
        '}\n'
    ),
    "loop_123": (
        '#include "loop_123.h"\n'
        '#include <algorithm>\n\n'
        'extern "C" void inner_loop_123(struct loop_123_data *data) {\n'
        '    std::sort(data->data, data->data + data->n);\n'
        '}\n'
    ),
    "loop_124": (
        '#include "loop_124.h"\n'
        '#include <algorithm>\n\n'
        'extern "C" void inner_loop_124(struct loop_124_data *data) {\n'
        '    std::sort(data->data, data->data + data->n);\n'
        '}\n'
    ),
    "loop_108": (
        '#include "loop_108.h"\n'
        '#include <stdint.h>\n\n'
        'extern "C" void inner_loop_108(struct loop_108_data *data) {\n'
        '    uint32_t *rgba = data->rgba;\n'
        '    uint8_t *y = data->y;\n'
        '    int64_t n = data->n;\n'
        '    for (int64_t i = 0; i < n; i++) {\n'
        '        y[i] = (rgba[i] >> 24) >> 2;\n'
        '        y[i] += ((rgba[i] >> 16) & 0xff) >> 1;\n'
        '        y[i] += ((rgba[i] >> 16) & 0xff) >> 3;\n'
        '        y[i] += ((rgba[i] >> 8) & 0xff) >> 3;\n'
        '    }\n'
        '}\n'
    ),
    # Recursive cascade_summation helper chain in loops/loop_105.c can't be
    # auto-extracted; the cascade result equals the total sum, so accumulate in
    # double (matches the float64 Python reference). `b` scratch is unused here.
    "loop_105": (
        '#include "loop_105.h"\n\n'
        'extern "C" void inner_loop_105(struct loop_105_data *data) {\n'
        '    float *a = data->a;\n'
        '    int n = data->n;\n'
        '    double res = 0.0;\n'
        '    for (int i = 0; i < n; i++) res += (double)a[i];\n'
        '    data->res = (float)res;\n'
        '}\n'
    ),
}


# Hand-written references for non-trivial loops.
_CUSTOM_REFS: dict[str, str] = {
    # ── Array-output loops ────────────────────────────────────────────────────
    "loop_027": (
        "import numpy as np\n\n"
        "def run(input):\n"
        "    return np.sqrt(input.astype(np.float64)).astype(np.float32)\n"
    ),
    "loop_028": (
        "import numpy as np\n\n"
        "def run(input1, input2):\n"
        "    return input1 / input2\n"
    ),
    "loop_029": (
        "import numpy as np\n\n"
        "def run(input, scale):\n"
        "    return np.ldexp(input, scale.astype(np.int32))\n"
    ),
    "loop_035": (
        "import numpy as np\n\n"
        "def run(a, b):\n"
        "    return a + b\n"
    ),
    "loop_113": (
        # Processes pairs (i += 2): c[2k]=a[2k]+a[2k+1], c[2k+1]=b[2k]+b[2k+1].
        # Kernel always reads a[i+1]/b[i+1] — wrap_inputs pads with 2 zeros so
        # OOB reads return 0. Reference mirrors that: pad then vectorize.
        "import numpy as np\n\n"
        "def run(a0, b0):\n"
        "    n = len(a0)\n"
        "    ap = np.concatenate([a0.astype(np.int32), np.zeros(2, np.int32)])\n"
        "    bp = np.concatenate([b0.astype(np.int32), np.zeros(2, np.int32)])\n"
        "    n_pairs = (n + 1) // 2\n"
        "    even = np.arange(n_pairs) * 2\n"
        "    c = np.zeros(n, dtype=np.int32)\n"
        "    # c[2k] = a[2k]+a[2k+1]\n"
        "    c_ev = (ap[even].astype(np.int64) + ap[even + 1].astype(np.int64)).astype(np.int32)\n"
        "    valid_ev = even[even < n]\n"
        "    c[valid_ev] = c_ev[:len(valid_ev)]\n"
        "    # c[2k+1] = b[2k]+b[2k+1]\n"
        "    c_od = (bp[even].astype(np.int64) + bp[even + 1].astype(np.int64)).astype(np.int32)\n"
        "    odd = even + 1\n"
        "    valid_od = odd[odd < n]\n"
        "    c[valid_od] = c_od[:len(valid_od)]\n"
        "    return c\n"
    ),
    "loop_108": (
        # Deinterleave RGBA uint32 → luminance-ish uint8.
        # rgba is stored as uint32 (see _DTYPE_OVERRIDES) so bitwise shifts work correctly.
        "import numpy as np\n\n"
        "def run(rgba):\n"
        "    u = rgba.view(np.uint32)\n"
        "    y = (u >> 26).astype(np.uint16)\n"           # (alpha >> 2)  = rgba >> 26
        "    y += ((u >> 17) & np.uint32(0x7f)).astype(np.uint16)\n"   # red >> 1
        "    y += ((u >> 19) & np.uint32(0x1f)).astype(np.uint16)\n"   # red >> 3
        "    y += ((u >> 11) & np.uint32(0x1f)).astype(np.uint16)\n"   # green >> 3
        "    return y.astype(np.uint8)\n"
    ),
    "loop_128": (
        "import numpy as np\n\n"
        "def run(a, b):\n"
        "    return a + b\n"
    ),
    # ── In-place sort loops ───────────────────────────────────────────────────
    "loop_120": "import numpy as np\n\ndef run(data):\n    return np.sort(data)\n",
    "loop_121": "import numpy as np\n\ndef run(data):\n    return np.sort(data)\n",
    "loop_122": "import numpy as np\n\ndef run(data):\n    return np.sort(data)\n",
    # ── Scalar-output loops ───────────────────────────────────────────────────
    "loop_105": (
        # Cascade (balanced pairwise) summation of `a`. `b` is a scratch buffer
        # the kernel overwrites — passed as a 2nd "input" by the generic scalar
        # ABI, ignored by the reference. n is always a power-of-2 >= 16.
        "import numpy as np\n\n"
        "def run(a, b):\n"
        "    return np.float32(np.sum(a.astype(np.float64)))\n"
    ),
    "loop_010": (
        "import numpy as np\n\n"
        "def run(a):\n"
        "    any_neg = bool(np.any(a < 0))\n"
        "    all_neg = bool(np.all(a < 0))\n"
        "    return np.int32(1 if all_neg else (2 if any_neg else 3))\n"
    ),
    "loop_032": (
        "import numpy as np\n\n"
        "def run(a, b):\n"
        "    lw = 0\n"
        "    res = np.float64(0.0)\n"
        "    for j in range(4, len(b), 5):\n"
        "        res -= a[lw] * b[j]\n"
        "        lw += 1\n"
        "    return np.float64(res)\n"
    ),
    "loop_126": (
        "import numpy as np\n\n"
        "def run(a, b):\n"
        "    res = np.uint32(0)\n"
        "    for i in range(len(a)):\n"
        "        res = np.uint32(int(res) + int(a[i]) * int(b[i]))\n"
        "        if res % 2:\n"
        "            res = np.uint32(int(res) + 1)\n"
        "    return res\n"
    ),
    "loop_127": (
        "import numpy as np\n\n"
        "def run(a, b):\n"
        "    # early exit on a[i]==512 never fires with generated inputs (values 1-100)\n"
        "    res = np.uint32(0)\n"
        "    for i in range(len(a)):\n"
        "        res = np.uint32(int(res) + int(a[i]) * int(b[i]))\n"
        "        if a[i] == 512:\n"
        "            break\n"
        "    return res\n"
    ),
}


def _gen_reference(info: LoopInfo) -> str:
    if info.loop_id in _CUSTOM_REFS:
        return _CUSTOM_REFS[info.loop_id]

    if info.is_inplace:
        f = info.input_ptr_fields[0]
        return f"import numpy as np\n\ndef run({f.name}):\n    return np.sort({f.name})\n"

    if info.is_array_output:
        # Generic fallback for array-output: element-wise add of first two inputs
        input_names = [f.name for f in info.input_ptr_fields]
        args = ", ".join(input_names)
        if len(input_names) == 1:
            body = f"    return {input_names[0]}.copy()"
        else:
            body = f"    return {input_names[0]} + {input_names[1]}"
        return f"import numpy as np\n\ndef run({args}):\n{body}\n"

    input_names = [f.name for f in info.ptr_fields]
    res_np = info.res_field.np_result_expr
    args = ", ".join(input_names)

    n_ptrs = len(info.ptr_fields)
    ct = info.ptr_fields[0].c_type  # dominant C type

    if n_ptrs == 1:
        a = input_names[0]
        if ct == "double":
            body = f"    return np.float64(np.sum({a}))"
        else:
            body = f"    return {res_np}(np.sum({a}))"
    else:
        a, b = input_names[0], input_names[1]
        if ct in ("float",):
            body = (f"    return np.float32(\n"
                    f"        np.dot({a}.astype(np.float64), {b}.astype(np.float64)))")
        elif ct in ("double",):
            body = f"    return np.float64(np.dot({a}, {b}))"
        elif ct in ("uint8_t",):
            # sum of abs diffs
            body = (f"    return np.uint32(\n"
                    f"        np.sum(np.abs({a}.astype(np.int32) - {b}.astype(np.int32)),\n"
                    f"               dtype=np.uint64))")
        elif ct in ("uint32_t", "int32_t", "int"):
            body = (f"    return {res_np}(\n"
                    f"        np.sum({a}.astype(np.uint64) * {b}.astype(np.uint64),\n"
                    f"               dtype=np.uint64))")
        elif ct in ("uint64_t", "int64_t"):
            body = (f"    return {res_np}(\n"
                    f"        np.sum({a}.astype(np.uint64) * {b}.astype(np.uint64),\n"
                    f"               dtype=np.uint64))")
        else:
            body = f"    return {res_np}(np.dot({a}, {b}))"

    return f"import numpy as np\n\ndef run({args}):\n{body}\n"


# ── Scalar solution kernel source ─────────────────────────────────────────────

def _extract_scalar_kernel(loop_id: str) -> str:
    """Extract the HAVE_AUTOVEC scalar function from loops/loop_NNN.c."""
    loop_num = re.search(r'loop_(\d+)', loop_id).group(1)
    c_file = LOOPS_DIR / f"{loop_id}.c"
    if not c_file.exists():
        return ""
    src = c_file.read_text()
    # Grab HAVE_AUTOVEC block
    m = re.search(
        r'#elif\s+\([^)]*HAVE_AUTOVEC[^)]*\).*?\n(.*?)(?=#elif|#else|#endif)',
        src, re.DOTALL
    )
    if not m:
        m = re.search(
            r'#elif\s+defined\(HAVE_AUTOVEC\).*?\n(.*?)(?=#elif|#else|#endif)',
            src, re.DOTALL
        )
    if not m:
        return ""
    code = m.group(1).strip()
    # Remove 'static' keyword
    code = re.sub(r'\bstatic\b\s*', '', code)
    # Remove 'restrict' and '__restrict__' (C99 — not valid in C++ mode)
    code = re.sub(r'\brestrict\b', '', code)
    code = re.sub(r'\b__restrict__\b', '', code)
    # Add extern "C" before void inner_loop
    code = re.sub(r'^void\s+inner_loop', 'extern "C" void inner_loop', code, flags=re.MULTILINE)
    return code


# ── Writers ───────────────────────────────────────────────────────────────────

def _stable_uuid(loop_id: str, n: int, source: str) -> str:
    return _uuid.uuid5(_uuid.NAMESPACE_DNS, f"{loop_id}_N{n}_{source}").hex


def _write_harness(info: LoopInfo) -> None:
    HARNESS_DIR.mkdir(parents=True, exist_ok=True)
    h_path   = HARNESS_DIR / f"{info.loop_id}.h"
    cpp_path = HARNESS_DIR / f"{info.loop_id}.cpp"

    h_content   = _gen_harness_h(info)
    cpp_content = _gen_harness_cpp(info)

    # Only write if content changed (avoid spurious git diffs)
    if not h_path.exists() or h_path.read_text() != h_content:
        h_path.write_text(h_content)
        print(f"  wrote {h_path.relative_to(REPO)}")

    if not cpp_path.exists() or cpp_path.read_text() != cpp_content:
        cpp_path.write_text(cpp_content)
        print(f"  wrote {cpp_path.relative_to(REPO)}")


def _write_definition(info: LoopInfo) -> None:
    out_dir = BENCH_TRACE / "definitions" / "simd-loop"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{info.loop_id}.json"

    dtype_ovr = _DTYPE_OVERRIDES.get(info.loop_id, {})
    inputs = {}
    for f in info.input_ptr_fields:
        dtype = dtype_ovr.get(f.name, f.numpy_dtype)
        inputs[f.name] = {"shape": ["N"], "dtype": dtype}

    override = _LOOP_META_OVERRIDES.get(info.loop_id, {})
    simd_loop_meta = {
        "output_inplace": info.is_inplace,
        "array_pad": override.get("array_pad", 0),
        "scratch": [{"name": f.name, "dtype": f.numpy_dtype} for f in info.scratch_fields],
    }

    if info.is_inplace:
        f = info.input_ptr_fields[0]
        dtype = _array_dtype_map_local.get(f.c_type, "int32")
        # Output must have a different name from the input to satisfy Definition validation
        out_name = f"sorted_{f.name}"
        out_spec = {"shape": ["N"], "dtype": dtype, "description": "Sorted array (in-place)"}
        definition = {
            "name": info.loop_id,
            "op_type": info.loop_id,
            "description": _description(info),
            "tags": ["simd-loop"],
            "axes": {"N": {"type": "var", "description": "Array length"}},
            "inputs": inputs,
            "outputs": {out_name: out_spec},
            "reference": _gen_reference(info),
            "simd_loop_meta": simd_loop_meta,
        }
        content = json.dumps(definition, indent=2) + "\n"
        if not out_path.exists() or out_path.read_text() != content:
            out_path.write_text(content)
            print(f"  wrote {out_path.relative_to(REPO)}")
        return

    _scalar_dtype_map = {
        "float": "float32", "double": "float64",
        "int": "int32", "int32_t": "int32", "int64_t": "int64",
        "uint32_t": "uint32", "uint64_t": "uint64",
        "uint8_t": "uint8", "uint16_t": "uint16",
        "int8_t": "int8", "int16_t": "int16",
        "bool": "int32",
    }

    if info.is_array_output:
        out_name = info.output_ptr.name
        out_dtype = _array_dtype_map_local.get(info.output_ptr.c_type, "int32")
        out_spec = {"shape": ["N"], "dtype": out_dtype, "description": "Output array"}
    else:
        out_name = info.res_field.name
        out_dtype = _scalar_dtype_map.get(info.res_field.c_type, "int32")
        out_spec = {"shape": None, "dtype": out_dtype, "description": "Scalar result"}

    definition = {
        "name": info.loop_id,
        "op_type": info.loop_id,
        "description": _description(info),
        "tags": ["simd-loop"],
        "axes": {"N": {"type": "var", "description": "Array length"}},
        "inputs": inputs,
        "outputs": {out_name: out_spec},
        "reference": _gen_reference(info),
        "simd_loop_meta": simd_loop_meta,
    }
    content = json.dumps(definition, indent=2) + "\n"
    if not out_path.exists() or out_path.read_text() != content:
        out_path.write_text(content)
        print(f"  wrote {out_path.relative_to(REPO)}")


def _description(info: LoopInfo) -> str:
    prob_dir = next((d for d in PROBLEMS_DIR.iterdir()
                     if d.name.startswith(info.loop_id + "_")), None)
    if prob_dir:
        import importlib.util
        spec = importlib.util.spec_from_file_location("prob", prob_dir / "problem.py")
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        meta = getattr(mod, "METADATA", {})
        return meta.get("description", meta.get("name", info.loop_id))
    return info.loop_id


def _write_workloads(info: LoopInfo) -> None:
    out_dir = BENCH_TRACE / "workloads" / "simd-loop"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{info.loop_id}.jsonl"

    wl_inputs = {f.name: {"type": "random"} for f in info.input_ptr_fields}
    lines = []
    for n in info.edge_sizes:
        uid = _stable_uuid(info.loop_id, n, "edge")
        lines.append(json.dumps({"axes": {"N": n}, "inputs": wl_inputs,
                                  "uuid": uid, "tags": {"source": "edge"}}))
    for n in info.perf_sizes:
        uid = _stable_uuid(info.loop_id, n, "perf")
        lines.append(json.dumps({"axes": {"N": n}, "inputs": wl_inputs,
                                  "uuid": uid, "tags": {"source": "perf"}}))

    content = "\n".join(lines) + "\n"
    if not out_path.exists() or out_path.read_text() != content:
        out_path.write_text(content)
        print(f"  wrote {out_path.relative_to(REPO)}")


def _write_reference_solution(info: LoopInfo) -> None:
    lid = info.loop_id
    out_dir = BENCH_TRACE / "solutions" / "simd-loop" / "reference-scalar" / lid
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"reference-scalar_{lid}.json"

    # 1. Custom scalar kernel (e.g. sort loops that can't use extracted code)
    if lid in _CUSTOM_SCALAR_KERNELS:
        scalar_src = _CUSTOM_SCALAR_KERNELS[lid]
    else:
        # 2. Extract from loops/loop_NNN.c
        scalar_src = _extract_scalar_kernel(lid)
        if scalar_src:
            scalar_src = f'#include "{lid}.h"\n#include <stdint.h>\n\n' + scalar_src + "\n"
        else:
            # 3. Fallback stub
            fallback_field = (info.res_field.name if info.res_field else
                              info.output_ptr.name if info.output_ptr else
                              info.input_ptr_fields[0].name if info.input_ptr_fields else "?")
            scalar_src = (
                f'#include "{lid}.h"\n'
                f'#include <stdint.h>\n\n'
                f'extern "C" void inner_{lid}(struct {lid}_data *data) {{\n'
                f'    // TODO: implement scalar reference\n'
                f'}}\n'
            )

    solution = {
        "name": f"reference-scalar_{lid}",
        "definition": lid,
        "dataset": "simd-loop",
        "author": "reference-scalar",
        "spec": {
            "language": "cpp",
            "target_hardware": ["aarch64"],
            "entry_point": f"kernel.cpp::inner_{lid}",
            "dependencies": [],
            "isa_features": [],
            "compile_flags": ["-O2", "-std=c++14"],
            "link_flags": [],
        },
        "sources": [
            {"path": f"{lid}.h",   "content": _gen_harness_h(info)},
            {"path": f"{lid}.cpp", "content": _gen_harness_cpp(info)},
            {"path": "kernel.cpp", "content": scalar_src},
        ],
        "description": f"Scalar reference for {lid}. Baseline for speedup measurement.",
    }
    content = json.dumps(solution, indent=2) + "\n"
    if not out_path.exists() or out_path.read_text() != content:
        out_path.write_text(content)
        print(f"  wrote {out_path.relative_to(REPO)}")


# ── Per-loop meta overrides ────────────────────────────────────────────────────
# Extra keys merged into the _LOOP_META entry for specific loops.
# Use "array_pad": N to request N extra trailing elements in all arrays for a
# loop whose kernel reads/writes beyond `size` (e.g. pair-processing loops).

_LOOP_META_OVERRIDES: dict[str, dict] = {
    "loop_113": {"array_pad": 2},  # kernel does a[i+1]/b[i+1] — needs 2 extra elements
}

# Per-loop input dtype overrides.  Use when a field's C type maps to int32 by
# default (to avoid sign issues with make_weights) but the kernel actually needs
# the full unsigned range.
_DTYPE_OVERRIDES: dict[str, dict[str, str]] = {
    "loop_108": {"rgba": "uint32"},  # pixel values need high bytes set; int32 range gives all-zero output
}


# ── Multi-axis loops (m/n/k): GEMV / matmul / transpose / conv ─────────────────
# These have >1 var axis, so the generic single-N _classify path doesn't apply.
# Each entry declares, explicitly:
#   axes    — ordered axis names (also the int64 ABI arg order)
#   inputs  — input ptr field -> list of axis names giving its (multi-dim) shape
#   output  — (output ptr field, list of axis names giving its shape)
#   reference — Python reference source (receives inputs as numpy arrays)
#   sizes   — {"edge": [{axis: val, ...}, ...], "perf": [...]} workload points
# C types are read from the parsed STRUCT_DEF, so dtypes stay in sync with the loop.
# ABI: armbench_entry_loop_NNN(in1, ..., int64_t <axes...>, void* res_out)

@dataclass
class MultiAxisInfo:
    loop_id:   str
    loop_num:  str
    fields:    List[Field]            # parsed struct fields, original order
    axes:      List[str]              # ordered axis names (ABI int64 order)
    inputs:    dict                   # name -> [axis names]
    output:    tuple                  # (name, [axis names])
    reference: str
    sizes:     dict                   # {"edge": [...], "perf": [...]}
    scalar:    Optional[str] = None   # explicit scalar kernel; None → extract from loops/*.c

    def field(self, name: str) -> Field:
        return next(f for f in self.fields if f.name == name)


_MULTI_AXIS: dict[str, dict] = {
    "loop_223": {
        "axes":   ["m", "n"],
        "inputs": {"a": ["m", "n"]},
        "output": ("at", ["n", "m"]),
        "reference": (
            "import numpy as np\n\n"
            "def run(a):\n"
            "    return np.ascontiguousarray(a.T)\n"
        ),
        "sizes": {
            "edge": [{"m": 1, "n": 1}, {"m": 2, "n": 3}, {"m": 8, "n": 8},
                     {"m": 17, "n": 15}, {"m": 33, "n": 4}],
            "perf": [{"m": 256, "n": 256}],
        },
        # Real inner_loop_223 lives under `#if defined(HAVE_AUTOVEC)...` (not #elif),
        # which the generic extractor misses — supply it directly.
        "scalar": (
            '#include "loop_223.h"\n'
            '#include <stdint.h>\n\n'
            'extern "C" void inner_loop_223(struct loop_223_data *data) {\n'
            '    uint64_t m = data->m;\n'
            '    uint64_t n = data->n;\n'
            '    uint32_t *a = data->a;\n'
            '    uint32_t *at = data->at;\n'
            '    for (uint64_t i = 0; i < m; i++)\n'
            '        for (uint64_t j = 0; j < n; j++)\n'
            '            at[j * m + i] = a[i * n + j];\n'
            '}\n'
        ),
    },
    "loop_220": {
        # Row-major GEMV: b[i] = sum_j a[i*n+j] * x[j]  ==  A @ x
        "axes":   ["m", "n"],
        "inputs": {"a": ["m", "n"], "x": ["n"]},
        "output": ("b", ["m"]),
        "reference": (
            "import numpy as np\n\n"
            "def run(a, x):\n"
            "    return (a.astype(np.float64) @ x.astype(np.float64)).astype(np.float32)\n"
        ),
        "sizes": {
            "edge": [{"m": 1, "n": 1}, {"m": 3, "n": 5}, {"m": 8, "n": 8},
                     {"m": 17, "n": 15}, {"m": 4, "n": 33}],
            "perf": [{"m": 256, "n": 256}],
        },
        "scalar": (
            '#include "loop_220.h"\n'
            '#include <stdint.h>\n\n'
            'extern "C" void inner_loop_220(struct loop_220_data *data) {\n'
            '    uint64_t m = data->m;\n'
            '    uint64_t n = data->n;\n'
            '    float *a = data->a;\n'
            '    float *x = data->x;\n'
            '    float *b = data->b;\n'
            '    for (uint64_t i = 0; i < m; i++) {\n'
            '        float d = 0;\n'
            '        for (uint64_t j = 0; j < n; j++) d += a[(i * n) + j] * x[j];\n'
            '        b[i] = d;\n'
            '    }\n'
            '}\n'
        ),
    },
    "loop_221": {
        # Row-major fp64 GEMV: b[i] = sum_j a[i*n+j] * x[j]  ==  A @ x
        "axes":   ["m", "n"],
        "inputs": {"a": ["m", "n"], "x": ["n"]},
        "output": ("b", ["m"]),
        "reference": (
            "import numpy as np\n\n"
            "def run(a, x):\n"
            "    return a.astype(np.float64) @ x.astype(np.float64)\n"
        ),
        "sizes": {
            "edge": [{"m": 1, "n": 1}, {"m": 3, "n": 5}, {"m": 8, "n": 8},
                     {"m": 17, "n": 15}, {"m": 4, "n": 33}],
            "perf": [{"m": 256, "n": 256}],
        },
        "scalar": (
            '#include "loop_221.h"\n'
            '#include <stdint.h>\n\n'
            'extern "C" void inner_loop_221(struct loop_221_data *data) {\n'
            '    uint64_t m = data->m;\n'
            '    uint64_t n = data->n;\n'
            '    double *a = data->a;\n'
            '    double *x = data->x;\n'
            '    double *b = data->b;\n'
            '    for (uint64_t i = 0; i < m; i++) {\n'
            '        double d = 0;\n'
            '        for (uint64_t j = 0; j < n; j++) d += a[(i * n) + j] * x[j];\n'
            '        b[i] = d;\n'
            '    }\n'
            '}\n'
        ),
    },
    "loop_216": {
        # Col-major fp32 GEMV: b[i] = sum_j a[m*j+i] * x[j]. Storing `a` as an
        # (n, m) C-order array makes flat[m*j+i] == a[j, i], so b == a.T @ x.
        "axes":   ["m", "n"],
        "inputs": {"a": ["n", "m"], "x": ["n"]},
        "output": ("b", ["m"]),
        "reference": (
            "import numpy as np\n\n"
            "def run(a, x):\n"
            "    return (a.astype(np.float64).T @ x.astype(np.float64)).astype(np.float32)\n"
        ),
        "sizes": {
            "edge": [{"m": 1, "n": 1}, {"m": 3, "n": 5}, {"m": 8, "n": 8},
                     {"m": 17, "n": 15}, {"m": 4, "n": 33}],
            "perf": [{"m": 256, "n": 256}],
        },
        "scalar": (
            '#include "loop_216.h"\n'
            '#include <stdint.h>\n\n'
            'extern "C" void inner_loop_216(struct loop_216_data *data) {\n'
            '    uint64_t m = data->m;\n'
            '    uint64_t n = data->n;\n'
            '    float *a = data->a;\n'
            '    float *x = data->x;\n'
            '    float *b = data->b;\n'
            '    for (uint64_t i = 0; i < m; i++) {\n'
            '        float d = 0;\n'
            '        for (uint64_t j = 0; j < n; j++) d += a[(m * j) + i] * x[j];\n'
            '        b[i] = d;\n'
            '    }\n'
            '}\n'
        ),
    },
    "loop_218": {
        # Col-major fp64 GEMV: b[i] = sum_j a[m*j+i] * x[j]  ==  a.T @ x  (a is (n,m)).
        "axes":   ["m", "n"],
        "inputs": {"a": ["n", "m"], "x": ["n"]},
        "output": ("b", ["m"]),
        "reference": (
            "import numpy as np\n\n"
            "def run(a, x):\n"
            "    return a.astype(np.float64).T @ x.astype(np.float64)\n"
        ),
        "sizes": {
            "edge": [{"m": 1, "n": 1}, {"m": 3, "n": 5}, {"m": 8, "n": 8},
                     {"m": 17, "n": 15}, {"m": 4, "n": 33}],
            "perf": [{"m": 256, "n": 256}],
        },
        "scalar": (
            '#include "loop_218.h"\n'
            '#include <stdint.h>\n\n'
            'extern "C" void inner_loop_218(struct loop_218_data *data) {\n'
            '    uint64_t m = data->m;\n'
            '    uint64_t n = data->n;\n'
            '    double *a = data->a;\n'
            '    double *x = data->x;\n'
            '    double *b = data->b;\n'
            '    for (uint64_t i = 0; i < m; i++) {\n'
            '        double d = 0;\n'
            '        for (uint64_t j = 0; j < n; j++) d += a[(m * j) + i] * x[j];\n'
            '        b[i] = d;\n'
            '    }\n'
            '}\n'
        ),
    },
    "loop_217": {
        # Row-major uint8 GEMV: c[y] = sum_x a[y*n+x] * b[x]  ==  a @ b, into uint32.
        "axes":   ["m", "n"],
        "inputs": {"a": ["m", "n"], "b": ["n"]},
        "output": ("c", ["m"]),
        "reference": (
            "import numpy as np\n\n"
            "def run(a, b):\n"
            "    return (a.astype(np.uint64) @ b.astype(np.uint64)).astype(np.uint32)\n"
        ),
        "sizes": {
            "edge": [{"m": 1, "n": 1}, {"m": 3, "n": 5}, {"m": 8, "n": 8},
                     {"m": 17, "n": 15}, {"m": 4, "n": 33}],
            "perf": [{"m": 256, "n": 256}],
        },
        "scalar": (
            '#include "loop_217.h"\n'
            '#include <stdint.h>\n\n'
            'extern "C" void inner_loop_217(struct loop_217_data *data) {\n'
            '    uint64_t m = data->m;\n'
            '    uint64_t n = data->n;\n'
            '    uint8_t *a = data->a;\n'
            '    uint8_t *b = data->b;\n'
            '    uint32_t *c = data->c;\n'
            '    for (uint64_t y = 0; y < m; y++) {\n'
            '        uint32_t d = 0;\n'
            '        for (uint64_t x = 0; x < n; x++) d += (uint32_t)a[y * n + x] * (uint32_t)b[x];\n'
            '        c[y] = d;\n'
            '    }\n'
            '}\n'
        ),
    },
    "loop_219": {
        # Col-major uint8 GEMV: c[y] = sum_x a[x*m+y] * b[x]  ==  a.T @ b (a is (n,m)), into uint32.
        "axes":   ["m", "n"],
        "inputs": {"a": ["n", "m"], "b": ["n"]},
        "output": ("c", ["m"]),
        "reference": (
            "import numpy as np\n\n"
            "def run(a, b):\n"
            "    return (a.astype(np.uint64).T @ b.astype(np.uint64)).astype(np.uint32)\n"
        ),
        "sizes": {
            "edge": [{"m": 1, "n": 1}, {"m": 3, "n": 5}, {"m": 8, "n": 8},
                     {"m": 17, "n": 15}, {"m": 4, "n": 33}],
            "perf": [{"m": 256, "n": 256}],
        },
        "scalar": (
            '#include "loop_219.h"\n'
            '#include <stdint.h>\n\n'
            'extern "C" void inner_loop_219(struct loop_219_data *data) {\n'
            '    uint64_t m = data->m;\n'
            '    uint64_t n = data->n;\n'
            '    uint8_t *a = data->a;\n'
            '    uint8_t *b = data->b;\n'
            '    uint32_t *c = data->c;\n'
            '    for (uint64_t y = 0; y < m; y++) {\n'
            '        uint32_t d = 0;\n'
            '        for (uint64_t x = 0; x < n; x++) d += (uint32_t)a[x * m + y] * (uint32_t)b[x];\n'
            '        c[y] = d;\n'
            '    }\n'
            '}\n'
        ),
    },
}


# ARM scalar typedefs (from <arm_neon.h>) → standard C types. The generated
# harness compiles standalone (no NEON headers), so normalize to plain types.
# float16_t / bfloat16_t are left as-is and handled per-loop when those land.
_NORM_CTYPE = {"float32_t": "float", "float64_t": "double"}


def _build_multi_axis_info(loop_id: str) -> Optional[MultiAxisInfo]:
    cfg = _MULTI_AXIS.get(loop_id)
    if cfg is None:
        return None
    loop_num = re.search(r'loop_(\d+)', loop_id).group(1)
    prob_dir = next((d for d in PROBLEMS_DIR.iterdir()
                     if d.name.startswith(loop_id + "_")), None)
    if not prob_dir:
        return None
    import importlib.util
    spec = importlib.util.spec_from_file_location("prob", prob_dir / "problem.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fields = _parse_struct(getattr(mod, "STRUCT_DEF", ""))
    for f in fields:
        f.c_type = _NORM_CTYPE.get(f.c_type, f.c_type)
    return MultiAxisInfo(
        loop_id=loop_id, loop_num=loop_num, fields=fields,
        axes=cfg["axes"], inputs=cfg["inputs"], output=cfg["output"],
        reference=cfg["reference"], sizes=cfg["sizes"], scalar=cfg.get("scalar"),
    )


def _gen_ma_harness_h(info: MultiAxisInfo) -> str:
    lid = info.loop_id
    struct_lines = []
    for f in info.fields:
        if f.is_ptr:
            struct_lines.append(f"    {f.c_type} *{f.name};")
        else:
            struct_lines.append(f"    {f.c_type} {f.name};")
    in_args = [f"void *{name}" for name in info.inputs]
    axis_args = [f"int64_t {a}" for a in info.axes]
    args = ", ".join(in_args + axis_args + ["void *res_out"])
    return f"""\
// Auto-generated by scripts/gen_simd_loop_harness.py — do not hand-edit.
#pragma once
#include <stdint.h>

struct {lid}_data {{
{chr(10).join(struct_lines)}
}};

#ifdef __cplusplus
extern "C" {{
#endif
int armbench_entry_{lid}({args});
#ifdef __cplusplus
}}
#endif
"""


def _gen_ma_harness_cpp(info: MultiAxisInfo) -> str:
    lid = info.loop_id
    in_args = [f"void *{name}" for name in info.inputs]
    axis_args = [f"int64_t {a}" for a in info.axes]
    args = ", ".join(in_args + axis_args + ["void *res_out"])

    assigns = []
    for name in info.inputs:
        ct = info.field(name).c_type
        assigns.append(f"    data.{name} = static_cast<{ct} *>({name});")
    out_name = info.output[0]
    out_ct = info.field(out_name).c_type
    assigns.append(f"    data.{out_name} = static_cast<{out_ct} *>(res_out);")
    for a in info.axes:
        ct = info.field(a).c_type
        assigns.append(f"    data.{a} = static_cast<{ct}>({a});")

    return f"""\
// Auto-generated by scripts/gen_simd_loop_harness.py — do not hand-edit.
#include "{lid}.h"
#include <stdint.h>

extern "C" void inner_{lid}(struct {lid}_data *data);

extern "C" int armbench_entry_{lid}({args}) {{
    struct {lid}_data data;
{chr(10).join(assigns)}
    inner_{lid}(&data);
    return 0;
}}
"""


def _write_multi_axis(info: MultiAxisInfo) -> None:
    lid = info.loop_id
    h_content   = _gen_ma_harness_h(info)
    cpp_content = _gen_ma_harness_cpp(info)

    # 1. On-disk harness copies (legacy fallback)
    HARNESS_DIR.mkdir(parents=True, exist_ok=True)
    for path, content in ((HARNESS_DIR / f"{lid}.h", h_content),
                          (HARNESS_DIR / f"{lid}.cpp", cpp_content)):
        if not path.exists() or path.read_text() != content:
            path.write_text(content)
            print(f"  wrote {path.relative_to(REPO)}")

    # 2. Definition
    inputs = {}
    for name, ax in info.inputs.items():
        dtype = _array_dtype_map_local.get(info.field(name).c_type, "int32")
        inputs[name] = {"shape": ax, "dtype": dtype}
    out_name, out_ax = info.output
    out_dtype = _array_dtype_map_local.get(info.field(out_name).c_type, "int32")
    definition = {
        "name": lid,
        "op_type": lid,
        "description": _description_ma(info),
        "tags": ["simd-loop"],
        "axes": {a: {"type": "var", "description": f"axis {a}"} for a in info.axes},
        "inputs": inputs,
        "outputs": {out_name: {"shape": out_ax, "dtype": out_dtype,
                               "description": "Output array"}},
        "reference": info.reference,
        "simd_loop_meta": {
            "output_inplace": False,
            "array_pad": 0,
            "scratch": [],
            "axes_order": info.axes,
        },
    }
    def_dir = BENCH_TRACE / "definitions" / "simd-loop"
    def_dir.mkdir(parents=True, exist_ok=True)
    def_path = def_dir / f"{lid}.json"
    content = json.dumps(definition, indent=2) + "\n"
    if not def_path.exists() or def_path.read_text() != content:
        def_path.write_text(content)
        print(f"  wrote {def_path.relative_to(REPO)}")

    # 3. Workloads
    wl_inputs = {name: {"type": "random"} for name in info.inputs}
    lines = []
    for source in ("edge", "perf"):
        for axvals in info.sizes.get(source, []):
            key = "_".join(f"{a}{axvals[a]}" for a in info.axes)
            uid = _uuid.uuid5(_uuid.NAMESPACE_DNS, f"{lid}_{key}_{source}").hex
            lines.append(json.dumps({"axes": axvals, "inputs": wl_inputs,
                                     "uuid": uid, "tags": {"source": source}}))
    wl_dir = BENCH_TRACE / "workloads" / "simd-loop"
    wl_dir.mkdir(parents=True, exist_ok=True)
    wl_path = wl_dir / f"{lid}.jsonl"
    content = "\n".join(lines) + "\n"
    if not wl_path.exists() or wl_path.read_text() != content:
        wl_path.write_text(content)
        print(f"  wrote {wl_path.relative_to(REPO)}")

    # 4. Reference-scalar solution. Prefer the explicit scalar kernel from the
    #    config; fall back to extracting from loops/loop_NNN.c.
    if info.scalar:
        scalar_src = info.scalar
    elif (extracted := _extract_scalar_kernel(lid)):
        scalar_src = f'#include "{lid}.h"\n#include <stdint.h>\n\n' + extracted + "\n"
    else:
        scalar_src = (f'#include "{lid}.h"\n#include <stdint.h>\n\n'
                      f'extern "C" void inner_{lid}(struct {lid}_data *data) {{\n'
                      f'    // TODO: implement scalar reference\n}}\n')
    solution = {
        "name": f"reference-scalar_{lid}",
        "definition": lid,
        "dataset": "simd-loop",
        "author": "reference-scalar",
        "spec": {
            "language": "cpp",
            "target_hardware": ["aarch64"],
            "entry_point": f"kernel.cpp::inner_{lid}",
            "dependencies": [],
            "isa_features": [],
            "compile_flags": ["-O2", "-std=c++14"],
            "link_flags": [],
        },
        "sources": [
            {"path": f"{lid}.h",   "content": _gen_ma_harness_h(info)},
            {"path": f"{lid}.cpp", "content": _gen_ma_harness_cpp(info)},
            {"path": "kernel.cpp", "content": scalar_src},
        ],
        "description": f"Scalar reference for {lid}. Baseline for speedup measurement.",
    }
    sol_dir = BENCH_TRACE / "solutions" / "simd-loop" / "reference-scalar" / lid
    sol_dir.mkdir(parents=True, exist_ok=True)
    sol_path = sol_dir / f"reference-scalar_{lid}.json"
    content = json.dumps(solution, indent=2) + "\n"
    if not sol_path.exists() or sol_path.read_text() != content:
        sol_path.write_text(content)
        print(f"  wrote {sol_path.relative_to(REPO)}")


def _description_ma(info: MultiAxisInfo) -> str:
    prob_dir = next((d for d in PROBLEMS_DIR.iterdir()
                     if d.name.startswith(info.loop_id + "_")), None)
    if prob_dir:
        import importlib.util
        spec = importlib.util.spec_from_file_location("prob", prob_dir / "problem.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        meta = getattr(mod, "METADATA", {})
        return meta.get("description", info.loop_id)
    return info.loop_id


# ── Main ──────────────────────────────────────────────────────────────────────

# Loops to process. Add more here as they are validated.
TARGET_LOOP_IDS = [
    # ── Scalar-output (reduction → single value) ─────────────────────────────
    "loop_001", "loop_002", "loop_003", "loop_004",  # fp32/uint32/fp64/uint64 inner products
    "loop_008",   # precise fp64 add reduction
    "loop_010",   # conditional reduction → int
    "loop_024",   # sum of abs diffs (uint8)
    "loop_032",   # fp64 banded dot product
    "loop_033",   # fp64 inner product (int64 n)
    "loop_126",   # conditional dot product
    "loop_127",   # dot product with early exit
    "loop_105",   # cascade pairwise summation (b is scratch; n is power-of-2 >= 16)
    # ── Array-output (element-wise transform → output array) ─────────────────
    "loop_027",   # fp32 sqrt  (1 input → 1 output)
    "loop_028",   # fp64 div   (2 inputs → 1 output)
    "loop_029",   # fp64 scalbn / ldexp (double + int64 scale → double)
    "loop_035",   # fp32 add   (2 inputs → 1 output)
    "loop_108",   # RGBA pixel deinterleave → uint8 luminance (LD4 + shift-accumulate)
    "loop_113",   # uint32 pair-wise add (2 inputs → 1 output, stride-2)
    "loop_128",   # uint32 add with aliased pointers
    # ── In-place sort (data sorted in-place, output IS data) ─────────────────
    "loop_120",   # insertion sort
    "loop_121",   # quicksort (with temp scratch buffer)
    "loop_122",   # odd-even transposition sort
    "loop_123",   # bitonic mergesort (temp + block_sizes scratch; n must be power-of-2)
    "loop_124",   # radix sort (temp + hist + prfx scratch)
]


def main() -> None:
    import sys
    dry_run = "--dry-run" in sys.argv

    all_infos: list[LoopInfo] = []

    for loop_id in TARGET_LOOP_IDS:
        # Find the problem directory
        prob_dirs = [d for d in PROBLEMS_DIR.iterdir()
                     if d.name.startswith(loop_id + "_")]
        if not prob_dirs:
            print(f"[skip] {loop_id}: problem dir not found")
            continue

        # Read STRUCT_DEF
        import importlib.util
        spec = importlib.util.spec_from_file_location("prob", prob_dirs[0] / "problem.py")
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        struct_def = getattr(mod, "STRUCT_DEF", "")

        fields = _parse_struct(struct_def)
        info   = _classify(loop_id, fields)

        if info is None:
            print(f"[skip] {loop_id}: struct pattern not supported")
            continue

        if info.is_inplace:
            scratch = [f.name for f in info.scratch_fields]
            print(f"[gen]  {loop_id} (inplace): data={info.input_ptr_fields[0].name}"
                  f" scratch={scratch} n={info.n_field.name}")
        elif info.is_array_output:
            out_desc = f"[]{info.output_ptr.name} ({info.output_ptr.c_type}[])"
            print(f"[gen]  {loop_id} (array): {[f.name for f in info.input_ptr_fields]} + "
                  f"{info.n_field.name} -> {out_desc}")
        else:
            print(f"[gen]  {loop_id}: {[f.name for f in info.ptr_fields]} + "
                  f"{info.n_field.name} -> {info.res_field.name} ({info.res_field.c_type})")

        if not dry_run:
            _write_harness(info)
            _write_definition(info)
            _write_workloads(info)
            _write_reference_solution(info)

        all_infos.append(info)

    # ── Multi-axis loops (m/n/k) ──────────────────────────────────────────────
    ma_count = 0
    for loop_id in _MULTI_AXIS:
        ma = _build_multi_axis_info(loop_id)
        if ma is None:
            print(f"[skip] {loop_id}: problem dir / struct not found")
            continue
        in_desc = {n: ax for n, ax in ma.inputs.items()}
        print(f"[gen]  {loop_id} (multi-axis {ma.axes}): {in_desc} -> "
              f"{ma.output[0]}{ma.output[1]}")
        if not dry_run:
            _write_multi_axis(ma)
        ma_count += 1

    total = len(all_infos) + ma_count
    if not dry_run and total:
        print(f"\nDone. Generated {len(all_infos)} single/array/inplace + "
              f"{ma_count} multi-axis loops.")
    elif dry_run:
        print(f"\nDry run: would generate {total} loops.")


if __name__ == "__main__":
    main()
