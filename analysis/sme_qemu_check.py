#!/usr/bin/env python3
"""Correctness check of the baseline-sme2 simd-loop kernels WITHOUT SME hardware:
build each solution natively (clang-18, -march=armv9-a+sve2+sme2) and run the
per-size probe under QEMU user-mode emulation (-cpu max implements SME; SME2
needs QEMU >= 9.1). Speed is meaningless here; only pass/fail matters.
  python analysis/sme_qemu_check.py [--qemu PATH] [--python PATH] [loop ids...]
"""
import json, os, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from bench.data.solution import Solution
from bench.data.definition import Definition
from bench.compile.builders.simd_loop import SimdLoopBuilder

def main():
    argv = sys.argv[1:]; qemu = "qemu-aarch64"; py = sys.executable
    while argv and argv[0] in ("--qemu", "--python"):
        if argv[0] == "--qemu": qemu = argv[1]
        else: py = argv[1]
        argv = argv[2:]
    only = {a if a.startswith("loop_") else f"loop_{a}" for a in argv}
    BT = ROOT / "bench-trace"; rows = []
    for sp in sorted((BT / "solutions/simd-loop/baseline-sme2").glob("*/baseline-sme2_*.json")):
        sol = Solution.model_validate(json.loads(sp.read_text())); lid = sol.definition
        if only and lid not in only: continue
        d = Definition.model_validate(json.loads((BT / "definitions/simd-loop" / f"{lid}.json").read_text()))
        try:
            so = SimdLoopBuilder().build(d, sol).so_path
        except Exception as e:
            msg = str(e); m = [l for l in msg.splitlines() if "error:" in l]
            rows.append((lid, "COMPILE_FAIL", (m[0] if m else msg)[:120])); print(rows[-1], flush=True); continue
        # small + medium + one perf-ish size per var axis (emulation is slow)
        wl = [json.loads(l)["axes"] for l in (BT / "workloads/simd-loop" / f"{lid}.jsonl").read_text().splitlines() if l.strip()]
        keep = [w for w in wl if max(w.values()) <= 200000][:4]
        cases = [",".join(f"{k}={v}" for k, v in w.items()) for w in keep]
        cmd = [qemu, "-cpu", "max", py, str(ROOT / "analysis/probe_simd_sizes.py"), lid, "--author", "baseline-sme2",
               "--so", str(so), "--axes", *cases]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        lines = [l for l in r.stdout.splitlines() if l.startswith("  {")]
        verdict = "PASS" if lines and all(": PASS" in l and "OVERRUN" not in l for l in lines) else "FAIL"
        if r.returncode != 0 and not lines: verdict = "CRASH"
        rows.append((lid, verdict, " | ".join(l.strip() for l in lines)[:300] + ("" if r.returncode == 0 else f" rc={r.returncode} " + r.stderr[-200:].replace("\n", " "))))
        print(rows[-1], flush=True)
    print("=== SUMMARY ===")
    for v in ("PASS", "FAIL", "CRASH", "COMPILE_FAIL"):
        print(v, [r[0] for r in rows if r[1] == v])
    json.dump([dict(loop=a, verdict=b, note=c) for a, b, c in rows], open(ROOT / "sme_qemu_check.json", "w"), indent=1)

if __name__ == "__main__":
    main()
