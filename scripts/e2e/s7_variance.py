#!/usr/bin/env python3
"""S7: agent run-to-run variance from repeated seeds of one identical cell.

Each S7 seed re-runs the E1 sve arm's exact protocol (c8g.xlarge on-demand, 40/50, simd-loop
pinned to baseline-sve2) with a distinct author/label/group, so the spread across seeds is the
benchmark's AGENT noise: same model, same prompt, same hardware, same references, different
sampling. Pair it with S9 (scripts/e2e/s9_table.py), which re-times fixed code and measures
0.40% geomean, to split agent noise from measurement noise.

Earlier numbers quoted seed 1 against the E1 sve arm. That is a valid second sample but the
two differ in bookkeeping; seed-vs-seed is the like-for-like comparison.

  python scripts/e2e/s7_variance.py            # all available seeds
  python scripts/e2e/s7_variance.py --seeds 1 2
"""
import argparse, collections, math, statistics
import wandb

PROJ = "ArmBench/arm-bench-kernels-gpt5.6-luna"
DS = ["ncnn", "llama.cpp", "simd-loop"]
G = "nanobot__gpt-5.6-luna__{ds}__sve__S7_seed{seed}"
g = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=PROJ)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    a = ap.parse_args()
    api = wandb.Api(timeout=120)

    per = collections.defaultdict(dict)
    counts = {}
    for s in a.seeds:
        n = 0
        for ds in DS:
            for r in api.runs(a.project, filters={"group": G.format(ds=ds, seed=s)}, per_page=100):
                v = r.summary.get("best_speedup") or r.summary.get("best_so_far") or 0.0
                if float(v) > 0:
                    per[(ds, r.name)][s] = float(v)
                    n += 1
        counts[s] = n
        print(f"seed {s}: {n} runs")
    seeds = [s for s in a.seeds if counts.get(s)]
    if len(seeds) < 2:
        raise SystemExit("\nneed at least two seeds with runs")

    paired = {k: v for k, v in per.items() if all(s in v for s in seeds)}
    print(f"\npaired across seeds {seeds}: {len(paired)} kernels\n")

    spreads = []
    for k, v in paired.items():
        xs = [v[s] for s in seeds]
        spreads.append((max(xs) - min(xs)) / statistics.mean(xs) * 100)
    geos = [g([v[s] for v in paired.values()]) for s in seeds]

    # Range grows with sample count, so (max-min)/mean is NOT comparable across different
    # numbers of seeds: the same population gives 1.38% at n=2 and 3.95% at n=3. Report the
    # sample standard deviation as well, which is what should be compared against anything.
    cvs = []
    for k, v in paired.items():
        xs = [v[s_] for s_ in seeds]
        cvs.append(statistics.stdev(xs) / statistics.mean(xs) * 100)
    gsd = statistics.stdev(geos) if len(geos) > 1 else float("nan")
    print(f"--- sample-size-independent (use these) ---")
    print(f"geomean-level sigma      : {gsd:.4f}  ({gsd / statistics.mean(geos) * 100:.2f}% CV)")
    print(f"median per-kernel CV     : {statistics.median(cvs):.2f}%")
    print(f"90th   per-kernel CV     : {sorted(cvs)[int(0.9 * len(cvs))]:.2f}%")
    print(f"--- range-based (n-dependent, do not compare across n) ---")
    print(f"median per-kernel spread : {statistics.median(spreads):.2f}%")
    print(f"mean   per-kernel spread : {statistics.mean(spreads):.2f}%")
    print(f"90th   per-kernel spread : {sorted(spreads)[int(0.9 * len(spreads))]:.2f}%")
    print(f"max    per-kernel spread : {max(spreads):.2f}%")
    print(f"geomean per seed         : " + ", ".join(f"{x:.4f}" for x in geos))
    print(f"geomean spread           : {(max(geos) - min(geos)) / statistics.mean(geos) * 100:.2f}%")

    print(f"\nworst 10 kernels by spread:")
    worst = sorted(paired, key=lambda k: -(max(paired[k][s] for s in seeds) - min(paired[k][s] for s in seeds))
                   / statistics.mean([paired[k][s] for s in seeds]))
    hdr = "".join(f"{'seed'+str(s):>9}" for s in seeds)
    print(f"{'kernel':<46}{hdr}{'spread':>9}")
    for k in worst[:10]:
        xs = [paired[k][s] for s in seeds]
        sp = (max(xs) - min(xs)) / statistics.mean(xs) * 100
        print(f"{k[0]+'/'+k[1]:<46}" + "".join(f"{x:>9.3f}" for x in xs) + f"{sp:>8.1f}%")


if __name__ == "__main__":
    main()
