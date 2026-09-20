#!/usr/bin/env python3
"""Kernel inventory for the end-to-end experiment: read a GGUF *header* and
report, per (tensor role, quant type, shape), how many layers use it and what
share of the per-token weight-byte traffic it carries (decode on a 4B model is
memory-bound, so byte share ~= time share for the mul_mat kernels).

Only the header is needed, so a partial download works:
    curl -L -r 0-67108863 -o q4km_head.bin \
      https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf
    python3 scripts/e2e/qwen35_inventory.py q4km_head.bin --json inventory.json

The JSON lists the unique mul_mat shapes (type, N, K) with their per-token
byte share; scripts/e2e/gen_qwen35_definitions.py consumes it.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import struct
import sys

# ggml type id -> (name, bits per weight) for the types a Q4_K_M/Q4_0 file uses.
GGML_TYPES = {
    0: ("F32", 32.0), 1: ("F16", 16.0), 2: ("Q4_0", 4.5), 3: ("Q4_1", 5.0),
    6: ("Q5_0", 5.5), 7: ("Q5_1", 6.0), 8: ("Q8_0", 8.5),
    10: ("Q2_K", 2.5625), 11: ("Q3_K", 3.4375), 12: ("Q4_K", 4.5),
    13: ("Q5_K", 5.5), 14: ("Q6_K", 6.5625), 30: ("BF16", 16.0),
}
# Which dataset gemm quant tag each ggml type maps to (None = no gemm definition).
DATASET_QUANT = {"Q4_K": "q4_k_m", "Q5_K": "q5_k", "Q6_K": "q6_k", "Q8_0": "q8_0"}


def read_header(path: str):
    f = open(path, "rb")
    u32 = lambda: struct.unpack("<I", f.read(4))[0]
    u64 = lambda: struct.unpack("<Q", f.read(8))[0]
    def s():
        n = u64(); return f.read(n).decode("utf-8", "replace")
    SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
    def val(t):
        if t == 8: return s()
        if t == 9:
            et = u32(); n = u64(); return [val(et) for _ in range(n)]
        return struct.unpack(FMT[t], f.read(SZ[t]))[0]
    if f.read(4) != b"GGUF":
        sys.exit(f"{path}: not a GGUF file")
    version = u32(); n_tensors = u64(); n_kv = u64()
    kv = {}
    for _ in range(n_kv):
        k = s(); t = u32(); v = val(t)
        kv[k] = f"[array len {len(v)}]" if isinstance(v, list) and len(v) > 16 else v
    tensors = []
    for _ in range(n_tensors):
        name = s(); nd = u32(); dims = [u64() for _ in range(nd)]; ty = u32(); off = u64()
        tensors.append((name, dims, ty, off))
    return version, kv, tensors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf", help="GGUF file (a partial download containing the header is enough)")
    ap.add_argument("--json", help="write the inventory JSON here")
    args = ap.parse_args()
    version, kv, tensors = read_header(args.gguf)
    arch = kv.get("general.architecture", "?")
    print(f"gguf v{version}  arch={arch}  tensors={len(tensors)}")
    for k in sorted(kv):
        if k.startswith(f"{arch}.") and not isinstance(kv[k], str) or k in ("general.name", "general.file_type"):
            print(f"  {k} = {kv[k]}")

    rows = collections.OrderedDict()
    for name, dims, ty, _ in tensors:
        role = re.sub(r"blk\.\d+\.", "blk.N.", name)
        tname, bpw = GGML_TYPES.get(ty, (str(ty), 0.0))
        key = (role, tname, tuple(dims))
        n = 1
        for d in dims: n *= d
        r = rows.setdefault(key, {"count": 0, "params": 0, "bytes": 0.0})
        r["count"] += 1; r["params"] += n; r["bytes"] += n * bpw / 8
    # Per-token traffic: every weight tensor is read once per token except
    # token_embd, which is read once as the tied lm_head (mul_mat) and once as
    # a single get_rows row (negligible). Count it once.
    total = sum(r["bytes"] for r in rows.values())
    print(f"\n{'n':>3} {'role':34} {'type':5} {'shape [K, N]':16} {'MB/token':>9} {'share':>6}")
    inv = []
    for (role, tname, dims), r in sorted(rows.items(), key=lambda kv: -kv[1]["bytes"]):
        share = r["bytes"] / total
        print(f"{r['count']:>3} {role:34} {tname:5} {str(list(dims)):16} {r['bytes']/1e6:9.1f} {share:6.1%}")
        if len(dims) == 2 and DATASET_QUANT.get(tname) and dims[0] >= 256 and dims[1] >= 32:
            inv.append({"role": role, "ggml_type": tname, "quant": DATASET_QUANT[tname],
                        "K": int(dims[0]), "N": int(dims[1]), "layers": r["count"],
                        "share_per_token": round(share, 4)})
    by_type = collections.Counter()
    for (_, tname, _), r in rows.items(): by_type[tname] += r["bytes"]
    print("\nper-token weight bytes by type: " + ", ".join(f"{t} {b/total:.1%}" for t, b in by_type.most_common()))
    print(f"total {total/1e6:.0f} MB/token")
    if args.json:
        json.dump({"arch": arch, "file": args.gguf, "total_bytes_per_token": total,
                   "mul_mat": inv}, open(args.json, "w"), indent=1)
        print(f"wrote {args.json} ({len(inv)} mul_mat shapes)")


if __name__ == "__main__":
    main()
