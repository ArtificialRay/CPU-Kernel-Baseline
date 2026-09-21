#!/usr/bin/env python3
"""Compile the agent's submitted kernels into shared objects and write the
override manifest that scripts/e2e/override/ loads at runtime.

For each definition directory under one or more agent-run roots
(<root>/<definition>/trajectory.jsonl + v<N>.cpp, as bench_fleet leaves them),
take the version the agent SUBMITTED (the trajectory's submit record's
source_file; --best picks the best PASSED evaluate instead), compile it exactly
like the harness did — the reference-scalar harness files (gemm.h/gemm.cpp)
from bench-trace + the agent's kernel.cpp, -O3 + the ISA's -march — and emit
  manifest.json: {"kernels":[{"op":"mul_mat","type":"q4_K","K":2560,"N":9216,
                               "so":".../gemm_ggml_q4_K_n9216_k2560.so",
                               "symbol":"armbench_entry_gemm","abi":"entry",
                               "definition":..., "version":..., "speedup":...}]}
Only definitions named gemm_ggml_<type>_n<N>_k<K> are spliceable (packed ABI);
others are listed under "skipped" with a reason.

  python3 scripts/e2e/build_manifest.py --runs agent-runs-e2e-sve2 --isa sve2 --out ~/e2e/manifest.json
  --only-types q4_K   (attribution runs: restrict to one quant type)
  --only-roles ffn    (substring match on definition name)
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from contracts import _isa_table  # noqa: E402  (config/kernel_contracts.yaml isa table)

NAME_RX = re.compile(r"^gemm_ggml_(q[456]_K)_n(\d+)_k(\d+)$")


def pick_version(traj: Path, best: bool):
    recs = [json.loads(l) for l in traj.read_text().splitlines() if l.strip()]
    by_version = {}
    for r in recs:
        if r.get("tool") == "compile" and r.get("source_file"):
            by_version[int(r["metrics"].get("version", 0))] = r["source_file"]
    if best:
        cands = [(r["metrics"].get("time_speedup_geomean") or 0, r["turn"]) for r in recs
                 if r.get("tool") == "evaluate" and r["metrics"].get("status") == "PASSED"]
        if not cands:
            return None, None
        # evaluate follows the compile of the same version: last compile before that turn
        _, turn = max(cands)
        prev = [r for r in recs if r.get("tool") == "compile" and r["turn"] < turn and r.get("source_file")]
        r = prev[-1]
        return r["source_file"], max(cands)[0]
    sub = [r for r in recs if "submit" in (r.get("tool") or "")]
    if not sub:
        return None, None
    r = sub[-1]
    return r.get("source_file") or r["metrics"].get("source_file"), r["metrics"].get("time_speedup_geomean") or r["metrics"].get("time_speedup")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="agent-run roots (later roots override earlier)")
    ap.add_argument("--isa", default="sve2")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--best", action="store_true", help="best PASSED evaluate instead of the submitted version")
    ap.add_argument("--only-types", nargs="*", default=None)
    ap.add_argument("--only-roles", nargs="*", default=None)
    ap.add_argument("--cxx", default="clang++-18")
    args = ap.parse_args()
    out_dir = args.out.parent.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    march = _isa_table()[args.isa].march

    found = {}
    for root in args.runs:
        for d in sorted(Path(root).expanduser().iterdir()):
            if (d / "trajectory.jsonl").exists():
                found[d.name] = d
    kernels, skipped = [], []
    for name, d in sorted(found.items()):
        m = NAME_RX.match(name)
        if not m:
            skipped.append({"definition": name, "reason": "not a packed-ABI gemm definition"}); continue
        qtype, N, K = m.group(1), int(m.group(2)), int(m.group(3))
        if args.only_types and qtype not in args.only_types:
            skipped.append({"definition": name, "reason": "filtered by --only-types"}); continue
        if args.only_roles and not any(r in name for r in args.only_roles):
            skipped.append({"definition": name, "reason": "filtered by --only-roles"}); continue
        src_file, speedup = pick_version(d / "trajectory.jsonl", args.best)
        if not src_file:
            skipped.append({"definition": name, "reason": "no submitted/PASSED version"}); continue
        ref = json.load(open(REPO / "bench-trace/solutions/llama.cpp/reference-scalar/gemm" / f"{name}.json"))
        build = out_dir / "build" / name
        build.mkdir(parents=True, exist_ok=True)
        for s in ref["sources"]:
            if s["path"] != "kernel.cpp":
                (build / s["path"]).write_text(s["content"])
        (build / "kernel.cpp").write_text((d / src_file).read_text())
        so = out_dir / f"{name}.so"
        cmd = [args.cxx, "-shared", "-fPIC", "-O3", march, "-std=c++14", "-I", str(build),
               str(build / "kernel.cpp"), str(build / "gemm.cpp"), "-o", str(so)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            skipped.append({"definition": name, "reason": "compile failed: " + p.stderr[-400:]}); continue
        kernels.append({"op": "mul_mat", "type": qtype, "K": K, "N": N, "so": str(so),
                        "symbol": "armbench_entry_gemm", "abi": "entry",
                        "definition": name, "version": src_file, "harness_speedup": speedup})
        print(f"built {name} <- {src_file} (harness speedup {speedup})")
    args.out.write_text(json.dumps({"isa": args.isa, "march": march, "kernels": kernels, "skipped": skipped}, indent=1))
    print(f"{len(kernels)} kernels, {len(skipped)} skipped -> {args.out}")


if __name__ == "__main__":
    main()
