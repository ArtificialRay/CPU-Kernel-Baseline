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
what closes that hole; see e2e/README.md.

Producing the dumps (llama.cpp built with e2e/adapters/llamacpp/override/apply.sh):

    ARMBENCH_OVERRIDES=<manifest.json> ARMBENCH_DUMP_DIR=<dir> \\
    ARMBENCH_DUMP_CALLS=4 ARMBENCH_DUMP_STRIDE=8 \\
    llama-perplexity -m <gguf> -f wiki.test.raw --chunks 1 -c 512 -ub 512 --no-warmup

--no-warmup matters: without it llama-perplexity's 1-2 token warmup pass is the first
captured call of every shape. Dumps with fewer than --min-rows rows are skipped for
that reason.

ARMBENCH_DUMP_STRIDE spreads the captures over the layer stack (the same (type,K,N) shape
recurs once per layer and early/late layers differ a lot). Then:

    capture_activations(dump_dir, "Qwen3.5-4B-Q4_K_M / wikitext-2 test chunk 0")      # or: python -m e2e prepare --steps capture --dump ... --source ...
"""
from __future__ import annotations

import json
import re
import uuid as _uuid
from pathlib import Path

import numpy as np

from e2e.errors import StepFailed

REPO = Path(__file__).resolve().parents[3]

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


def capture_activations(dump: Path, source: str, root: Path = REPO / "bench-trace", *,
                        min_rows: int = 8, dry_run: bool = False) -> int:
    """Rewrite the A inputs of the published gemm_ggml_* workloads to real activations.

    dump      ARMBENCH_DUMP_DIR contents (from a llama.cpp built with the override hook)
    source    provenance string recorded on every generated workload input, e.g.
              'Qwen3.5-4B-Q4_K_M / wikitext-2 test chunk 0'
    min_rows  ignore captured calls with fewer rows (llama-perplexity warmup)
    Returns how many workload inputs were rewritten.
    """
    dumps = find_dumps(dump)
    if not dumps:
        raise StepFailed(f"no *.call*.a.bin under {dump}")
    for key, calls in list(dumps.items()):
        small = [c for c in calls if c[2] is not None and c[2] < min_rows]
        if small:
            print(f"  {key}: dropping {len(small)} call(s) with M < {min_rows} "
                  f"(warmup? re-dump with --no-warmup)")
            dumps[key] = [c for c in calls if c not in small]
            if not dumps[key]:
                del dumps[key]
    wl_dir = root / "workloads" / "gemm"
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
        tdir = root / "tensors" / "gemm" / name
        if not dry_run:
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
            if not dry_run:
                np.save(root / rel, sl)
            w["inputs"]["A"] = {
                "type": "tensor",
                "path": rel,
                "source": f"{source}; {dm['type']} K={K} N={dm['N']} call {call}, rows {start}:{start+M}",
            }
            w["uuid"] = _uuid.uuid5(_uuid.NAMESPACE_URL, f"armbench/{name}/realact/M{M}/{rel}").hex
            w.setdefault("tags", {})["inputs"] = "real-activations"
            written += 1
        if not dry_run:
            wl_path.write_text("".join(json.dumps(w) + "\n" for w in rows))
        print(f"{name}: {len(rows)} workloads -> real activations from {len(avail)} captured call(s)")

    print(f"\n{written} workload inputs rewritten; {skipped} definitions had no dump")
    return written
