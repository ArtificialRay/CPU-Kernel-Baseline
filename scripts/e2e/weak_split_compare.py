#!/usr/bin/env python3
"""Reconcile two definitions of a "weak" expert baseline.

The deck (and the paper's Sec 4.3 TODO) splits weak from strong by KERNEL FAMILY:
"ncnn has a strong baseline in conv2d kernels, while other kernel baseline is weak;
llama.cpp has a strong baseline in all quantized kernel, while the full-precision
kernel is weak". Our analysis splits by a MEASURED per-kernel threshold,
`baseline_vs_scalar < 2.0` (analysis/wandb_log_run.py), i.e. the expert baseline is
less than 2x faster than naive scalar code.

Both are meant to answer "does the agent only win where the maintainer wasn't
looking?", and they disagree on magnitude (the deck reports 1.1-3.6x on the strong
set; we measure 0.690x). This prints them against the same runs so the disagreement
can be attributed to the definition rather than argued about.

  python scripts/e2e/weak_split_compare.py --arm sve2
"""
import argparse, collections, math
import wandb

PROJ = "ArmBench/arm-bench-kernels-gpt5.6-luna"
ARMS = {"neon": ["nanobot__gpt-5.6-luna__{ds}__neon__E1_c8g"],
        "sve": ["nanobot__gpt-5.6-luna__{ds}__sve__E1_c8g"],
        "sve2": ["nanobot__gpt-5.6-luna__{ds}__sve2__S2a",
                 "nanobot__gpt-5.6-luna__{ds}__sve2__pilot",
                 "nanobot__gpt-5.6-luna__{ds}__sve2__E1_c8g"]}
DS = ["ncnn", "llama.cpp", "simd-loop"]
QUANT = ("q4_", "q5_", "q6_", "q8_", "_q4", "_q5", "_q6", "_q8", "w8a8")
g = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def deck_strong(ds, name):
    """The deck's family rule. Returns True (strong), False (weak), or None (unstated)."""
    n = name.lower()
    if ds == "ncnn":
        # "strong baseline in conv2d kernels" -- but the deck's own case study calls
        # deconv2d_depthwise "a very weak baseline", so deconv is excluded.
        return n.startswith("conv2d") or ("conv2d" in n and "deconv" not in n)
    if ds == "llama.cpp":
        return any(q in n for q in QUANT)
    return None  # the deck says nothing about SIMD Loops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=PROJ)
    ap.add_argument("--arm", default="sve2", choices=list(ARMS))
    ap.add_argument("--threshold", type=float, default=2.0)
    a = ap.parse_args()

    api = wandb.Api(timeout=120)
    rows = {}
    for ds in DS:
        for p in ARMS[a.arm]:
            for r in api.runs(a.project, filters={"group": p.format(ds=ds)}, per_page=100):
                sp = r.summary.get("best_speedup") or r.summary.get("best_so_far") or 0.0
                bvs = r.summary.get("baseline_vs_scalar")
                if float(sp) <= 0:
                    continue
                rows[(ds, r.name)] = (float(sp), None if bvs is None else float(bvs))

    missing = [k for k, (_, b) in rows.items() if b is None]
    print(f"arm={a.arm}  runs with a speedup: {len(rows)}   missing baseline_vs_scalar: {len(missing)}")
    if missing:
        print("  (excluded from the measured split):", [f"{d}/{n}" for d, n in missing][:6])

    meas = {k: (b >= a.threshold) for k, (_, b) in rows.items() if b is not None}
    deck = {k: deck_strong(k[0], k[1]) for k in rows}

    print(f"\n--- measured split (baseline_vs_scalar >= {a.threshold} = strong) ---")
    for lab, want in (("weak", False), ("strong", True)):
        ks = [k for k, v in meas.items() if v is want]
        print(f"  {lab:<7} n={len(ks):<3} geomean {g([rows[k][0] for k in ks]):.3f}")

    print("\n--- deck's family split (ncnn conv2d / llama.cpp quantized = strong) ---")
    for lab, want in (("weak", False), ("strong", True), ("unstated (simd-loop)", None)):
        ks = [k for k, v in deck.items() if v is want]
        print(f"  {lab:<20} n={len(ks):<3} geomean {g([rows[k][0] for k in ks]):.3f}")

    both = [k for k in rows if k in meas and deck[k] is not None]
    print(f"\n--- where the two rules disagree (n={len(both)} kernels the deck rules on) ---")
    cm = collections.Counter((meas[k], deck[k]) for k in both)
    print(f"  {'':<22}{'deck weak':>12}{'deck strong':>13}")
    for mv, lab in ((False, "measured weak"), (True, "measured strong")):
        print(f"  {lab:<22}{cm.get((mv, False), 0):>12}{cm.get((mv, True), 0):>13}")
    dis = [k for k in both if meas[k] != deck[k]]
    if dis:
        print("\n  disagreeing kernels (speedup, baseline_vs_scalar, measured/deck):")
        for k in sorted(dis, key=lambda k: rows[k][1]):
            sp, b = rows[k]
            print(f"    {k[0]+'/'+k[1]:<50} {sp:>7.3f}  bvs={b:>8.2f}  "
                  f"{'strong' if meas[k] else 'weak':<6}/{'strong' if deck[k] else 'weak'}")


if __name__ == "__main__":
    main()
