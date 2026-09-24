#!/usr/bin/env python3
"""S9: measurement-only variance.

speed_compare.py is re-run N times against FIXED code (autovec candidate,
baseline-sve2 reference) on one box. Nothing about the agent varies, so the
spread across repeats is the harness's own timing noise. Subtracting it from
S7's agent-resampling spread says how much of the run-to-run variation people
see in this benchmark is the model and how much is the stopwatch.

  python scripts/e2e/s9_table.py sweep_backups/s9/s9_rep*.json
"""
import json, math, statistics, sys


def load(paths):
    tab = {}
    for p in paths:
        for e in json.load(open(p))["results"]:
            tab.setdefault(e["definition"], []).append(e["speedup"])
    n = len(paths)
    missing = [d for d, v in tab.items() if len(v) != n]
    if missing:
        raise SystemExit(f"definitions missing from some repeats: {missing}")
    return tab


def main(paths):
    tab = load(paths)
    n = len(paths)
    hdr = "".join(f"{'rep'+str(i+1):>9}" for i in range(n))
    print(f"{'definition':<11}{hdr}{'spread%':>9}{'cv%':>7}")
    spreads, cvs = [], []
    for d, v in tab.items():
        mean = statistics.mean(v)
        spread = (max(v) - min(v)) / mean * 100
        cv = statistics.stdev(v) / mean * 100
        spreads.append(spread)
        cvs.append(cv)
        print(f"{d:<11}" + "".join(f"{x:9.4f}" for x in v) + f"{spread:9.2f}{cv:7.2f}")

    geos = [math.exp(statistics.mean(math.log(tab[d][i]) for d in tab)) for i in range(n)]
    worst = max(tab, key=lambda d: (max(tab[d]) - min(tab[d])) / statistics.mean(tab[d]))
    print()
    print(f"median per-kernel spread : {statistics.median(spreads):.2f}%")
    print(f"mean   per-kernel spread : {statistics.mean(spreads):.2f}%")
    print(f"worst  per-kernel spread : {max(spreads):.2f}%  ({worst})")
    print(f"median per-kernel CV     : {statistics.median(cvs):.2f}%")
    print(f"geomean per repeat       : " + ", ".join(f"{g:.4f}x" for g in geos))
    print(f"geomean spread           : {(max(geos) - min(geos)) / statistics.mean(geos) * 100:.2f}%")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
