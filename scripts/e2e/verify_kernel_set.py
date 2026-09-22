#!/usr/bin/env python3
"""Guard: prove a manifest actually contains the kernels you think it does.

A manifest built from hand-assembled solution files can silently be the *old*
kernels -- the plumbing still verifies (11/11 shapes intercepted, every gate
passes) because nothing about the splice is wrong; it is faithfully running the
wrong sources.  That costs a whole measurement run to notice, and only if you
happen to recognise the perplexity.

So before provisioning, check the manifest's sources against a reference set:

  --expect-changed <def>   this definition MUST differ from the reference
  --expect-same <def>      this definition MUST be byte-identical (deliberately reused)
  --forbid <regex>         no kernel's source may match (e.g. a known-bad idiom)

  python3 scripts/e2e/verify_kernel_set.py --manifest m.json \
      --reference ~/runs-orig --expect-changed gemm_ggml_q4_K_n4096_k2560 ... \
      --forbid 'maxabs >> sh'

Exits non-zero, loudly, on any mismatch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.e2e.build_manifest import kernel_rows_abi_text, pick_version  # noqa: E402


def sha(b: bytes) -> str:
    """Hash of the kernel source as the compiler would see it under the rows ABI.

    build_manifest leaves a *rewritten* kernel.cpp in its build dir (static scratch
    made thread_local), so raw run-dir sources never match it byte for byte.  The
    rewrite is idempotent, so normalising both sides through it compares like for
    like whether or not --rows-abi was used."""
    return hashlib.sha256(kernel_rows_abi_text(b.decode("utf-8", "replace")).encode()).hexdigest()[:12]


def manifest_sources(man: Path) -> dict:
    """definition -> (label, source bytes) for every kernel the manifest names.

    build_manifest records the version it compiled and leaves the exact
    kernel.cpp it fed the compiler under <out_dir>/build/<definition>/, which is
    the only copy guaranteed to be what actually went into the .so.
    """
    d = json.load(open(man))
    out = {}
    for k in d["kernels"]:
        built = Path(k["so"]).parent / "build" / k["definition"] / "kernel.cpp"
        if not built.exists():
            raise SystemExit(f"{k['definition']}: no build/kernel.cpp next to {k['so']}")
        out[k["definition"]] = (k.get("version", "?"), built.read_bytes())
    return out


def reference_sources(roots: list) -> dict:
    out = {}
    for root in roots:
        for d in sorted(Path(root).expanduser().iterdir()):
            traj = d / "trajectory.jsonl"
            if not traj.exists():
                continue
            src, _ = pick_version(traj, False)
            if src:
                out[d.name] = (src, (d / src).read_bytes())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--reference", nargs="*", default=[], help="agent-run roots holding the set being compared against")
    ap.add_argument("--expect-changed", nargs="*", default=[])
    ap.add_argument("--expect-same", nargs="*", default=[])
    ap.add_argument("--forbid", nargs="*", default=[])
    ap.add_argument("--expect-count", type=int, default=None)
    args = ap.parse_args()

    got = manifest_sources(args.manifest)
    ref = reference_sources(args.reference) if args.reference else {}
    fail = []

    if args.expect_count is not None and len(got) != args.expect_count:
        fail.append(f"manifest has {len(got)} kernels, expected {args.expect_count}")

    print(f"{'definition':<30} {'version':<10} {'sha':<14} {'vs reference'}")
    for name in sorted(got):
        ver, src = got[name]
        note = "(no reference)"
        if name in ref:
            note = "SAME as reference" if sha(src) == sha(ref[name][1]) else f"differs (ref {ref[name][0]} {sha(ref[name][1])})"
        print(f"{name:<30} {ver:<10} {sha(src):<14} {note}")

    for name in args.expect_changed:
        if name not in got:
            fail.append(f"{name}: expected in the manifest, absent")
        elif name not in ref:
            fail.append(f"{name}: no reference to compare against")
        elif sha(got[name][1]) == sha(ref[name][1]):
            fail.append(f"{name}: BYTE-IDENTICAL to the reference -- this is the old kernel, not the new one")
    for name in args.expect_same:
        if name not in got:
            fail.append(f"{name}: expected in the manifest, absent")
        elif name in ref and sha(got[name][1]) != sha(ref[name][1]):
            fail.append(f"{name}: expected to be reused unchanged but differs from the reference")
    for pat in args.forbid:
        rx = re.compile(pat)
        for name, (_, src) in sorted(got.items()):
            if rx.search(src.decode("utf-8", "replace")):
                fail.append(f"{name}: source matches forbidden pattern /{pat}/")

    if fail:
        print("\nFAILED:")
        for f in fail:
            print("  " + f)
        sys.exit(1)
    print(f"\nOK: {len(got)} kernels, {len(args.expect_changed)} verified new, "
          f"{len(args.expect_same)} verified reused, {len(args.forbid)} forbidden patterns absent")


if __name__ == "__main__":
    main()
