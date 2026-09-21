#!/usr/bin/env python3
"""Compile the reference-scalar (or any bench-trace author's) solutions for the
packed gemm definitions into override .so files + manifest, as a control for the
end-to-end splice: reference-scalar mirrors ggml's Q8_K per-256-block activation
quantization, so spliced into llama.cpp it must reproduce stock perplexity.

  python3 scripts/e2e/build_reference_manifest.py --author reference-scalar --isa sve2 --out ~/e2e/refscalar/manifest.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts" / "e2e"))
from contracts import _isa_table  # noqa: E402
from build_manifest import rows_abi_sources, NAME_RX  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--author", default="reference-scalar")
    ap.add_argument("--isa", default="sve2")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=None, help="definition names to include")
    ap.add_argument("--cxx", default="clang++-18")
    args = ap.parse_args()
    out_dir = args.out.parent.resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    march = _isa_table()[args.isa].march
    sol_dir = REPO / "bench-trace/solutions/llama.cpp" / args.author / "gemm"
    kernels, skipped = [], []
    for f in sorted(sol_dir.glob("gemm_ggml_*.json")):
        name = f.stem
        if args.only and name not in args.only:
            continue
        m = NAME_RX.match(name)
        if not m:
            continue
        qtype, N, K = m.group(1), int(m.group(2)), int(m.group(3))
        sol = json.load(open(f))
        build = out_dir / "build" / name; build.mkdir(parents=True, exist_ok=True)
        for s in sol["sources"]:
            (build / s["path"]).write_text(s["content"])
        rows_abi_sources(build, N)
        so = out_dir / f"{name}.so"
        cmd = [args.cxx, "-shared", "-fPIC", "-O3", march, "-std=c++14", "-I", str(build),
               str(build / "kernel.cpp"), str(build / "gemm.cpp"), "-o", str(so)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            skipped.append({"definition": name, "reason": p.stderr[-300:]}); print("FAILED", name); continue
        kernels.append({"op": "mul_mat", "type": qtype, "K": K, "N": N, "so": str(so), "symbol": "armbench_entry_gemm",
                        "abi": "entry_rows", "definition": name, "author": args.author})
        print("built", name)
    args.out.write_text(json.dumps({"isa": args.isa, "march": march, "author": args.author, "kernels": kernels, "skipped": skipped}, indent=1))
    print(f"{len(kernels)} kernels, {len(skipped)} skipped -> {args.out}")


if __name__ == "__main__":
    main()
