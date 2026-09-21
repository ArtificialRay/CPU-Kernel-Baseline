#!/usr/bin/env python3
"""Turn llama.cpp activation dumps into bench-trace workloads with real inputs.

Why this exists
---------------
The packed-ggml gemm definitions were evaluated on `{"type": "random"}` activations.
Random activations are near-Gaussian and zero-mean; a real post-norm hidden state has a
handful of channels one to two orders of magnitude larger than the rest, and per-32-element
block sums that are nowhere near zero. Both of the numerical shortcuts the 2026-09-21 Fable
kernels took -- one activation scale per row instead of per 256-element block, and an int8
requantization of the block sums under one exponent shared across the batch -- are invisible
on random data (0.3 dB) and cost 3.9 perplexity in the model. Gating on real activations is
what closes that hole; see docs/e2e_qwen35.md.

Producing the dumps (llama.cpp built with scripts/e2e/override/apply.sh):

    ARMBENCH_OVERRIDES=<manifest.json> ARMBENCH_DUMP_DIR=<dir> \\
    ARMBENCH_DUMP_CALLS=4 ARMBENCH_DUMP_STRIDE=8 \\
    llama-perplexity -m <gguf> -f wiki.test.raw --chunks 1 -c 512 -ub 512

ARMBENCH_DUMP_STRIDE spreads the captures over the layer stack (the same (type,K,N) shape
recurs once per layer and early/late layers differ a lot). Then:

    python scripts/e2e/dump_to_workloads.py --dump <dir> --root bench-trace
"""
from __future__ import annotations

import argparse
import json
import re
import uuid as _uuid
from pathlib import Path

import numpy as np

DUMP_RX = re.compile(r"^(?P<type>q\d_[K0]|q8_0)_K(?P<K>\d+)_N(?P<N>\d+)\.call(?P<call>\d+)\.a\.bin$")
DEF_RX = re.compile(r"^gemm_ggml_(?P<type>q\d_[K0]|q8_0)_n(?P<N>\d+)_k(?P<K>\d+)$")


def find_dumps(dump_dir: Path):
    """(type, K, N) -> [(call_index, path, M)] sorted by call index."""
    out = {}
    for p in sorted(dump_dir.glob("*.call*.a.bin")):
        m = DUMP_RX.match(p.name)
        if not m:
            continue
        meta = p.with_suffix(".meta")
        M = json.loads(meta.read_text())["M"] if meta.exists() else None
        key = (m["type"], int(m["K"]), int(m["N"]))
        out.setdefault(key, []).append((int(m["call"]), p, M))
    for v in out.values():
        v.sort()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True, help="ARMBENCH_DUMP_DIR contents")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2] / "bench-trace")
    ap.add_argument("--source", default="Qwen3.5-4B-Q4_K_M / wikitext-2 test chunk 0",
                    help="provenance string recorded on every generated workload input")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dumps = find_dumps(args.dump)
    if not dumps:
        print(f"no *.call*.a.bin under {args.dump}")
        return 1
    wl_dir = args.root / "workloads" / "gemm"
    written = skipped = 0

    for wl_path in sorted(wl_dir.glob("gemm_ggml_*.jsonl")):
        name = wl_path.stem
        dm = DEF_RX.match(name)
        if not dm:
            continue
        key = (dm["type"], int(dm["K"]), int(dm["N"]))
        avail = dumps.get(key)
        if not avail:
            skipped += 1
            continue
        K = int(dm["K"])
        rows = [json.loads(l) for l in wl_path.read_text().splitlines() if l.strip()]
        tdir = args.root / "tensors" / "gemm" / name
        if not args.dry_run:
            tdir.mkdir(parents=True, exist_ok=True)

        for i, w in enumerate(rows):
            M = int(w["axes"]["M"])
            call, src, dumped_M = avail[i % len(avail)]
            a = np.fromfile(src, dtype=np.uint16)
            total = a.size // K
            a = a.reshape(total, K)
            # A different window per workload so the M points are not nested prefixes
            # of one another (M=1 would otherwise always be the same token).
            start = (i * 37) % max(1, total - M + 1)
            sl = a[start:start + M]
            if sl.shape[0] != M:
                print(f"  {name} M={M}: dump only has {total} rows, skipping")
                continue
            rel = f"tensors/gemm/{name}/act_M{M}_call{call}_r{start}.npy"
            if not args.dry_run:
                np.save(args.root / rel, sl)
            w["inputs"]["A"] = {
                "type": "tensor",
                "path": rel,
                "source": f"{args.source}; {dm['type']} K={K} N={dm['N']} call {call}, rows {start}:{start+M}",
            }
            w["uuid"] = _uuid.uuid5(_uuid.NAMESPACE_URL, f"armbench/{name}/realact/M{M}/{rel}").hex
            w.setdefault("tags", {})["inputs"] = "real-activations"
            written += 1
        if not args.dry_run:
            wl_path.write_text("".join(json.dumps(w) + "\n" for w in rows))
        print(f"{name}: {len(rows)} workloads -> real activations from {len(avail)} captured call(s)")

    print(f"\n{written} workload inputs rewritten; {skipped} definitions had no dump")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
