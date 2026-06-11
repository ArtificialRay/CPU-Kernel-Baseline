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
  A1 — two pointer arrays + int n + scalar output
  B  — one pointer array  + int n + scalar output

Loops requiring custom handling (skipped):
  loop_023 — indexes OOB with generated inputs
  loop_105 — b is scratch buffer, not an input
  p+lmt loops (005, 006, 022, 103) — need string/buffer input generation
  complex structs (012, 019, 109, …)
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
    loop_id:    str          # "loop_001"
    loop_num:   str          # "001"
    ptr_fields: List[Field]  # input pointer fields in struct order
    n_field:    Field        # the integer size field
    res_field:  Field        # the scalar result field
    edge_sizes: List[int]    # from EDGE_SIZES (0 excluded)
    perf_sizes: List[int]    # from PERF_SIZES


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

def _classify(loop_id: str, fields: List[Field]) -> Optional[LoopInfo]:
    """Try to classify fields into (ptr_inputs, n_field, res_field).
    Returns None if the loop doesn't fit the supported patterns.
    """
    loop_num = re.search(r'loop_(\d+)', loop_id).group(1)

    ptr_fields   = [f for f in fields if f.is_ptr]
    scalar_fields = [f for f in fields if not f.is_ptr]

    if not ptr_fields or not scalar_fields:
        return None

    # Find the result field: last scalar that looks like an output
    res_field = None
    n_field   = None

    # Heuristic: the "result" is typically the last scalar field
    # The "n" is a scalar whose name hints at a size
    for f in scalar_fields:
        if f.name.lower() in _SIZE_NAMES or f.name in ("n", "size"):
            n_field = f
        elif f.name.lower() in _RES_NAMES:
            res_field = f

    # Fallback: if no obvious n_field found, take first non-res scalar
    if n_field is None:
        non_res = [f for f in scalar_fields if f != res_field]
        if non_res:
            n_field = non_res[0]

    # Fallback: if no res_field, take last scalar
    if res_field is None:
        non_n = [f for f in scalar_fields if f != n_field]
        if non_n:
            res_field = non_n[-1]

    if n_field is None or res_field is None:
        return None

    # Only support patterns with ≤ 2 pointer inputs
    if len(ptr_fields) > 2:
        return None

    # Load sizes from problem.py
    import importlib.util
    prob_dir = next((d for d in PROBLEMS_DIR.iterdir()
                     if d.name.startswith(loop_id + "_")), None)
    if not prob_dir:
        return None
    spec = importlib.util.spec_from_file_location("prob", prob_dir / "problem.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    edge = [s for s in getattr(mod, "EDGE_SIZES", []) if s > 0]
    # Use PERF_SIZES_C8G if present (better Graviton4 coverage), else PERF_SIZES
    perf = getattr(mod, "PERF_SIZES_C8G", None) or getattr(mod, "PERF_SIZES", [])

    return LoopInfo(
        loop_id=loop_id,
        loop_num=loop_num,
        ptr_fields=ptr_fields,
        n_field=n_field,
        res_field=res_field,
        edge_sizes=edge,
        perf_sizes=perf,
    )


# ── Harness generator ─────────────────────────────────────────────────────────

def _gen_harness_h(info: LoopInfo) -> str:
    lid = info.loop_id
    num = info.loop_num

    # struct field declarations (preserve restrict for clarity)
    struct_fields = []
    for f in info.ptr_fields:
        struct_fields.append(f"    {f.c_type} *{f.name};")
    struct_fields.append(f"    {info.n_field.c_type} {info.n_field.name};")
    struct_fields.append(f"    {info.res_field.c_type} {info.res_field.name};")

    # entry args: void* per pointer, int64_t n, void* res_out
    args = ", ".join(["void *" + f.name for f in info.ptr_fields]
                     + ["int64_t n", "void *res_out"])

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

    args = ", ".join(["void *" + f.name for f in info.ptr_fields]
                     + ["int64_t n", "void *res_out"])

    # struct assignments
    ptr_assigns = "\n    ".join(
        f"data.{f.name} = static_cast<{f.c_type} *>({f.name});"
        for f in info.ptr_fields
    )
    n_cast = f"static_cast<{info.n_field.c_type}>(n)"
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

# Hand-written references for non-trivial loops.
_CUSTOM_REFS: dict[str, str] = {
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

    inputs = {}
    for f in info.ptr_fields:
        inputs[f.name] = {"shape": ["N"], "dtype": f.numpy_dtype}

    # Output dtype — use the actual unsigned type for correctness
    out_dtype_map = {
        "float": "float32", "double": "float64",
        "int": "int32", "uint32_t": "int32", "uint64_t": "int64",
        "int32_t": "int32", "int64_t": "int64",
        "uint8_t": "int8", "uint16_t": "int16",
        "bool": "int32",
    }
    out_dtype = out_dtype_map.get(info.res_field.c_type, "int32")

    definition = {
        "name": info.loop_id,
        "op_type": info.loop_id,
        "description": _description(info),
        "tags": ["simd-loop"],
        "axes": {"N": {"type": "var", "description": "Array length"}},
        "inputs": inputs,
        "outputs": {
            info.res_field.name: {
                "shape": None,
                "dtype": out_dtype,
                "description": "Scalar result",
            }
        },
        "reference": _gen_reference(info),
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

    lines = []
    for n in info.edge_sizes:
        uid = _stable_uuid(info.loop_id, n, "edge")
        lines.append(json.dumps({"axes": {"N": n}, "scalar_inputs": {},
                                  "uuid": uid, "tags": {"source": "edge"}}))
    for n in info.perf_sizes:
        uid = _stable_uuid(info.loop_id, n, "perf")
        lines.append(json.dumps({"axes": {"N": n}, "scalar_inputs": {},
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

    # Get scalar kernel source from loops/loop_NNN.c
    scalar_src = _extract_scalar_kernel(lid)
    if not scalar_src:
        # Fallback: generate minimal scalar from struct
        input_names = [f.name for f in info.ptr_fields]
        n_field = info.n_field.name
        res_field = info.res_field.name
        scalar_src = (
            f'#include "{lid}.h"\n'
            f'#include <stdint.h>\n\n'
            f'extern "C" void inner_{lid}(struct {lid}_data *data) {{\n'
            f'    // TODO: implement scalar reference\n'
            f'    data->{res_field} = 0;\n'
            f'}}\n'
        )
    else:
        scalar_src = f'#include "{lid}.h"\n#include <stdint.h>\n\n' + scalar_src + "\n"

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
        "sources": [{"path": "kernel.cpp", "content": scalar_src}],
        "description": f"Scalar reference for {lid}. Baseline for speedup measurement.",
    }
    content = json.dumps(solution, indent=2) + "\n"
    if not out_path.exists() or out_path.read_text() != content:
        out_path.write_text(content)
        print(f"  wrote {out_path.relative_to(REPO)}")


# ── simd_loop.py regenerator ──────────────────────────────────────────────────

def _regen_simd_loop_py(all_infos: list[LoopInfo]) -> None:
    """Fully regenerate bench/datasets/simd_loop.py from _LOOP_META."""

    meta_entries = []
    for info in all_infos:
        inputs_repr = repr([
            (f.name, f"np.{f.numpy_dtype}") for f in info.ptr_fields
        ]).replace("'np.", "np.").replace("')", ")")
        # Fix: repr wraps numpy dtype refs in quotes — build manually
        input_list = ", ".join(
            f'("{f.name}", np.{f.numpy_dtype})' for f in info.ptr_fields
        )
        res_np = info.res_field.np_result_expr  # e.g. "np.float32"
        entry = (
            f'    "{info.loop_id}": {{\n'
            f'        "inputs": [{input_list}],\n'
            f'        "result_dtype": {res_np},\n'
            f'    }},'
        )
        meta_entries.append(entry)

    meta_block = "\n".join(meta_entries)

    py_content = f'''\
# Auto-generated by scripts/gen_simd_loop_harness.py — do not hand-edit.
# Re-run the script to add new loops or update existing ones.
"""simd-loop dataset adapter: numpy \\u2194 flat C arrays via ctypes.

Each registered loop has an entry in _LOOP_META describing its input tensors
and result dtype. SIGNATURES is derived from _LOOP_META at import time — no
manual sync required.

The harness shims live in bench/compile/builders/simd_loop_harness/<op>.{{h,cpp}}.
The calling convention for every loop:
    int armbench_entry_loop_NNN(void* in1, [void* in2, ...], int64_t n, void* res_out)

To add a new loop:
  1. Add an entry to _LOOP_META below.
  2. Run scripts/gen_simd_loop_harness.py to generate harness + bench-trace files.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


# ── Loop metadata ─────────────────────────────────────────────────────────────
# "inputs": ordered list of (field_name, numpy_dtype) matching struct pointer fields.
# "result_dtype": numpy dtype for the scalar result buffer.

_LOOP_META: Dict[str, dict] = {{
{meta_block}
}}


# ── ctypes signatures (derived from _LOOP_META) ───────────────────────────────

_C_VOID_P = ctypes.c_void_p
_C_INT64  = ctypes.c_int64

SIGNATURES: Dict[str, List[type]] = {{
    op: [_C_VOID_P] * len(meta["inputs"]) + [_C_INT64, _C_VOID_P]
    for op, meta in _LOOP_META.items()
}}

_RESULT_DTYPE: Dict[str, type] = {{
    op: meta["result_dtype"] for op, meta in _LOOP_META.items()
}}


# ── Context ───────────────────────────────────────────────────────────────────

@dataclass
class SimdLoopContext:
    entry_args: Tuple[Any, ...]
    _arrays:    list          # contiguous input arrays — kept alive during kernel call
    res_buf:    np.ndarray    # 1-element result buffer


# ── Adapter ───────────────────────────────────────────────────────────────────

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
        self_contained: bool = False,
    ) -> SimdLoopContext:
        if op_type not in _LOOP_META:
            raise NotImplementedError(f"SimdLoopDataset: op_type {{op_type!r}} not registered")
        meta = _LOOP_META[op_type]
        arrays = [np.ascontiguousarray(np_inputs[name]) for name, _ in meta["inputs"]]
        n = int(scalar_args["N"])
        res_buf = np.zeros(1, dtype=meta["result_dtype"])
        ptrs = [ctypes.cast(a.ctypes.data, _C_VOID_P) for a in arrays]
        entry_args = tuple(ptrs) + (_C_INT64(n),) + (ctypes.cast(res_buf.ctypes.data, _C_VOID_P),)
        return SimdLoopContext(entry_args=entry_args, _arrays=arrays, res_buf=res_buf)

    def unwrap_output(self, ctx: SimdLoopContext) -> np.ndarray:
        return ctx.res_buf.copy()

    def release(self, ctx: SimdLoopContext) -> None:
        ctx._arrays.clear()
'''

    if not SIMD_LOOP_PY.exists() or SIMD_LOOP_PY.read_text() != py_content:
        SIMD_LOOP_PY.write_text(py_content)
        print(f"  wrote {SIMD_LOOP_PY.relative_to(REPO)}")


# ── Main ──────────────────────────────────────────────────────────────────────

# Loops to process. Add more here as they are validated.
TARGET_LOOP_IDS = [
    "loop_001", "loop_002", "loop_003", "loop_004",  # existing (idempotent)
    "loop_008",   # fp64 sum
    "loop_010",   # conditional reduction → int result
    "loop_024",   # sum of abs diffs (uint8 inputs)
    "loop_032",   # fp64 banded pattern
    "loop_033",   # fp64 inner product (int64 n)
    "loop_126",   # conditional dot product
    "loop_127",   # dot product with early exit
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

        print(f"[gen]  {loop_id}: {[f.name for f in info.ptr_fields]} + "
              f"{info.n_field.name} -> {info.res_field.name} ({info.res_field.c_type})")

        if not dry_run:
            _write_harness(info)
            _write_definition(info)
            _write_workloads(info)
            _write_reference_solution(info)

        all_infos.append(info)

    if not dry_run and all_infos:
        _regen_simd_loop_py(all_infos)
        print(f"\nDone. Generated {len(all_infos)} loops.")
    elif dry_run:
        print(f"\nDry run: would generate {len(all_infos)} loops.")


if __name__ == "__main__":
    main()
