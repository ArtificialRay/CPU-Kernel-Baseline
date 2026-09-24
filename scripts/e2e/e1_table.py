#!/usr/bin/env python3
"""E1 three-arm ISA table (neon / sve / sve2) from W&B, matched on common kernels.

sve2 arm = the S2a runs (same protocol, same c8g, baseline-sve2), plus its pilot group.
Reports both the matched-set geomean and each arm's coverage, because a geomean over
"whatever that arm completed" silently rewards an arm that failed its hard kernels.
"""
import argparse, collections, math
import wandb

PROJ = "ArmBench/arm-bench-kernels-gpt5.6-luna"
# The sve2 arm is spread over three groups: the original S2a runs, its pilot, and the
# 2026-09-24 new-subset top-up, which a lane-script bug logged under __sve2__E1_c8g instead
# of __sve2__S2a. Read all three or the arm silently comes back 11 kernels short.
ARMS = {"neon": ["nanobot__gpt-5.6-luna__{ds}__neon__E1_c8g"],
        "sve": ["nanobot__gpt-5.6-luna__{ds}__sve__E1_c8g"],
        "sve2": ["nanobot__gpt-5.6-luna__{ds}__sve2__S2a",
                 "nanobot__gpt-5.6-luna__{ds}__sve2__pilot",
                 "nanobot__gpt-5.6-luna__{ds}__sve2__E1_c8g"]}
DS = ["ncnn", "llama.cpp", "simd-loop"]
g = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--project", default=PROJ); a = ap.parse_args()
    api = wandb.Api(timeout=120)
    best = collections.defaultdict(dict)
    for arm, pats in ARMS.items():
        for ds in DS:
            for p in pats:
                for r in api.runs(a.project, filters={"group": p.format(ds=ds)}, per_page=100):
                    v = r.summary.get("best_speedup") or r.summary.get("best_so_far") or 0.0
                    best[arm][(ds, r.name)] = float(v)
    for arm in ARMS:
        zeros = [k[1] for k, v in best[arm].items() if v == 0]
        print(f"{arm:<6} {len(best[arm]):>3} runs" + (f"   ZERO (compile-failed): {zeros}" if zeros else ""))
    common = set.intersection(*(set(best[a_]) for a_ in ARMS))
    scored = [k for k in common if all(best[a_][k] > 0 for a_ in ARMS)]
    print(f"\ncommon to all arms: {len(common)}   scored in all arms: {len(scored)}\n")
    print(f"{'dataset':<12}{'n':>4}   {'neon':>8}{'sve':>8}{'sve2':>8}")
    for ds in DS + ["ALL"]:
        ks = [k for k in scored if ds == "ALL" or k[0] == ds]
        print(f"{ds:<12}{len(ks):>4}   " + "".join(f"{g([best[a_][k] for k in ks]):>8.3f}" for a_ in ARMS))
    wins = collections.Counter(max(ARMS, key=lambda a_: best[a_][k]) for k in scored)
    print("\nper-kernel wins:", {a_: wins.get(a_, 0) for a_ in ARMS})
    print(f"\n{'kernel':<52}{'neon':>8}{'sve':>8}{'sve2':>8}")
    for k in sorted(scored):
        print(f"{k[0]+'/'+k[1]:<52}" + "".join(f"{best[a_][k]:>8.3f}" for a_ in ARMS))


if __name__ == "__main__":
    main()
