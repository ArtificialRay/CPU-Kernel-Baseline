#!/usr/bin/env python3
"""Perplexity across a quantization ladder, to give a kernel's quality cost a scale.

"Our kernels cost +1.0% perplexity" means nothing on its own. Measured against what
one step down the quantization ladder costs -- or what quantizing at all costs -- it
becomes a sentence a reader can judge. Same text, same chunk count, same threads,
same binary as the kernel measurement, so the numbers are directly comparable.

  python3 scripts/e2e/quant_reference.py --bin ~/llama.cpp-e2e/build-stock/bin/llama-perplexity \
      --repo unsloth/Qwen3.5-4B-GGUF --text ~/models/wiki.test.raw --chunks 8 --threads 16 \
      --files Qwen3.5-4B-Q4_K_M.gguf Qwen3.5-4B-Q5_K_M.gguf ... --out quantref.json

Downloads, measures and DELETES each file in turn (--keep to retain), so the whole
ladder needs only as much disk as its largest member. --reference names the file
everything is reported relative to.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

PPL_RX = re.compile(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)\s*\+/-\s*([0-9.]+)")


def fetch(repo: str, fname: str, dest: Path) -> bool:
    if dest.exists():
        return True
    url = f"https://huggingface.co/{repo}/resolve/main/{fname}"
    part = dest.with_suffix(dest.suffix + ".part")
    rc = subprocess.run(["curl", "-sSL", "-C", "-", "-o", str(part), url]).returncode
    if rc != 0:
        return False
    part.rename(dest)
    return True


def perplexity(binary: Path, model: Path, text: Path, chunks: int, threads: int):
    """Run llama-perplexity, echoing its progress as it goes.

    A bf16 model at 64 chunks takes two hours, and capturing the output whole means
    two hours of a log that says nothing -- indistinguishable from a hung process,
    which is exactly how it reads when you check on it.
    """
    proc = subprocess.Popen([str(binary), "-m", str(model), "-f", str(text),
                             "--chunks", str(chunks), "-t", str(threads)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines, last = [], 0.0
    for line in proc.stdout:
        lines.append(line)
        now = time.time()
        if "ETA" in line and now - last > 60:      # one heartbeat a minute, not one a chunk
            print(f"      {line.strip()[-90:]}", flush=True)
            last = now
    proc.wait(timeout=7200)
    out = "".join(lines)
    m = PPL_RX.search(out)
    if not m:
        return None, None, out[-300:].strip()
    return float(m.group(1)), float(m.group(2)), None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", type=Path, required=True, help="llama-perplexity binary")
    ap.add_argument("--repo", required=True, help="HF repo holding the GGUFs")
    ap.add_argument("--files", nargs="+", required=True, help="GGUF file names, easiest-to-hardest order")
    ap.add_argument("--text", type=Path, required=True)
    ap.add_argument("--models-dir", type=Path, default=Path.home() / "models")
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--reference", default=None, help="file everything is reported relative to (default: first)")
    ap.add_argument("--keep", action="store_true", help="keep each GGUF instead of deleting after measuring")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ref = args.reference or args.files[0]
    rows = []
    for fname in args.files:
        dest = args.models_dir / fname
        existed = dest.exists()
        print(f"[quantref] {fname}: fetching" if not existed else f"[quantref] {fname}: present", flush=True)
        if not fetch(args.repo, fname, dest):
            rows.append({"file": fname, "error": "download failed"})
            print(f"[quantref] {fname}: DOWNLOAD FAILED", flush=True)
            continue
        size_gb = dest.stat().st_size / 1e9
        ppl, err, fail = perplexity(args.bin, dest, args.text, args.chunks, args.threads)
        rows.append({"file": fname, "size_gb": round(size_gb, 2), "ppl": ppl, "stderr": err, "error": fail})
        print(f"[quantref] {fname}: PPL {ppl if ppl else fail}", flush=True)
        if not args.keep and not existed:
            dest.unlink(missing_ok=True)

    base = next((r["ppl"] for r in rows if r["file"] == ref and r.get("ppl")), None)
    print(f"\n{'model':<34} {'size':>7} {'PPL':>9} {'vs ' + ref.split('-')[-1].replace('.gguf',''):>12}")
    for r in rows:
        if not r.get("ppl"):
            print(f"{r['file']:<34} {'':>7} {'FAILED':>9}  {r.get('error','')[:40]}")
            continue
        rel = f"{(r['ppl'] / base - 1) * 100:+.2f}%" if base else "-"
        print(f"{r['file']:<34} {r['size_gb']:6.2f}G {r['ppl']:9.4f} {rel:>12}")
    if args.out:
        args.out.write_text(json.dumps({"repo": args.repo, "chunks": args.chunks, "threads": args.threads,
                                        "reference": ref, "rows": rows}, indent=1))
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
