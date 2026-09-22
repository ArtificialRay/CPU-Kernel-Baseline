#!/usr/bin/env python3
"""Repair the batch-shared-exponent block sums in the 2026-09-21 Fable kernels.

Every one of those kernels has two paths. At M == 1 it keeps the Q8_K activation block
sums exact, splitting each int16 into lo/hi bytes and recombining with {1,128,1,128}. At
M >= 2 it instead requantizes them to int8 under a single exponent `sh` derived from the
maximum over the whole batch, so the Q4_K/Q5_K min-correction term fits one `usmmla`.
That shared exponent is a batch-global scale on a per-row quantity, and on real weights
(where dmin runs ~8x d, so the min term carries most of the weight value) it costs 3.9
perplexity end to end. See docs/e2e_qwen35.md.

The repair keeps the batched i8mm path and makes the block sums exact: store lo and hi
(32 bytes per pair-block instead of 16), issue two `usmmla`, recombine with
`vmlaq_n_s32(mlo, mhi, 128)`, and drop the exponent. One extra multiply-accumulate per
(block, row-pair) out of roughly nine, and Q6_K/Q8_0 need nothing since they have no min
term.

    python scripts/e2e/patch_exact_bsums.py --author claude-code-claude-fable-5-1-sve2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

SUBS = [
    ("int32x4_t mm = vusmmlaq_s32(zero32, mn_op, vld1q_s8(bsq + so * 16));",
     "int32x4_t mlo = vusmmlaq_s32(zero32, mn_op, vld1q_s8(bsq + so * 32));\n"
     "                int32x4_t mhi = vusmmlaq_s32(zero32, mn_op, vld1q_s8(bsq + so * 32 + 16));\n"
     "                int32x4_t mm  = vmlaq_n_s32(mlo, mhi, 128);"),
    ("bsq.resize((size_t)P * K_blk * 16);", "bsq.resize((size_t)P * K_blk * 32);"),
    ("""                    int v = (bsr[j] + rnd) >> sh;
                    if (v > 127) v = 127;
                    if (v < -128) v = -128;
                    bsq[so * 16 + rr * 8 + j] = (int8_t)v;""",
     """                    const int v  = bsr[j];
                    const int lo = ((v + 64) & 127) - 64;
                    const int hi = (v - lo) >> 7;
                    bsq[so * 32 + rr * 8 + j]      = (int8_t)lo;
                    bsq[so * 32 + 16 + rr * 8 + j] = (int8_t)hi;"""),
    ("const float mscale = (float)(1 << sh);",
     "const float mscale = 1.0f;  // block sums are exact now (lo/hi split); no shared exponent"),
    ("const int8_t* bq = bsq.data() + (size_t)p0 * K_blk * 16;",
     "const int8_t* bq = bsq.data() + (size_t)p0 * K_blk * 32;"),
]

MARKER = "(maxabs >> sh)"   # the shared-exponent computation; absent means nothing to fix


def patch(src: str):
    if MARKER not in src:
        return None
    out = src
    for a, b in SUBS:
        if a not in out:
            raise ValueError(f"expected code not found, cannot patch safely: {a[:70]}...")
        out = out.replace(a, b)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=REPO / "bench-trace")
    ap.add_argument("--dataset", default="llama.cpp")
    ap.add_argument("--author", default="claude-code-claude-fable-5-1-sve2")
    ap.add_argument("--out-author", default="fable-exact-bsums")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src_dir = args.root / "solutions" / args.dataset / args.author / "gemm"
    out_dir = args.root / "solutions" / args.dataset / args.out_author / "gemm"
    if not src_dir.is_dir():
        print(f"no such author dir: {src_dir}")
        return 1
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    patched = copied = 0
    for p in sorted(src_dir.glob("*.json")):
        d = json.loads(p.read_text())
        ksrc = next(s for s in d["sources"] if s["path"] == "kernel.cpp")
        new = patch(ksrc["content"])
        if new is None:
            note = "no shared-exponent block sums; copied unchanged"
            copied += 1
        else:
            ksrc["content"] = new
            note = "block sums made exact (lo/hi split, two usmmla)"
            patched += 1
        d["author"] = args.out_author
        d["name"] = f"{args.out_author}_{d['definition']}"
        d["description"] = f"{args.author} kernel; {note}. See scripts/e2e/patch_exact_bsums.py."
        d.pop("provenance", None)
        print(f"{'PATCHED ' if new else 'copied  '}{d['definition']}")
        if not args.dry_run:
            (out_dir / p.name).write_text(json.dumps(d, indent=1))

    print(f"\n{patched} patched, {copied} copied unchanged -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
