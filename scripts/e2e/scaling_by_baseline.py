#!/usr/bin/env python3
"""Is the test-time-scaling plateau uniform, or concentrated where the expert left headroom?

WITHIN a single S1 run (100/110 tool-call budget): `performance_step_40` is best-so-far at
40 tool calls and `best_speedup` is best-so-far at the end, so their ratio is the gain from
the extra budget with hardware, baseline, model and seed all held fixed by construction.

An earlier version of this script compared the S1 runs against the E1 sve arm instead. That
was confounded and its result must not be used: S1 ran on c7g.xlarge (Graviton3) and E1 sve
on c8g.xlarge (Graviton4), and E1 additionally pins --baseline-author baseline-sve2 for
simd-loop while S1 takes the per-ISA default. Two independent runs also carry the 4.37%
geomean / 15.9% per-kernel agent resampling spread measured by S7, which within-run
truncation avoids entirely.

  python scripts/e2e/scaling_by_baseline.py
"""
import argparse, math, statistics
import wandb

PROJ = "ArmBench/arm-bench-kernels-gpt5.6-luna"
DS = ["ncnn", "llama.cpp", "simd-loop"]
S1 = "nanobot__gpt-5.6-luna__{ds}__sve__S1"
g = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=PROJ)
    ap.add_argument("--at", type=int, default=40, help="truncation point, in tool calls")
    a = ap.parse_args()
    api = wandb.Api(timeout=120)

    rows, skipped = [], 0
    for ds in DS:
        for r in api.runs(a.project, filters={"group": S1.format(ds=ds)}, per_page=100):
            s = r.summary
            end = s.get("best_speedup") or s.get("best_so_far") or 0.0
            weak = s.get("weak_baseline")
            if not end or weak is None:
                skipped += 1
                continue
            # best-so-far at <= 40 tool calls, reconstructed from the per-step history
            # (these runs predate the performance_step_40 summary field).
            hist = r.history(keys=["best_so_far"], pandas=False)
            pts = [h for h in hist if h.get("_step") is not None and h.get("best_so_far")]
            early = [h["best_so_far"] for h in pts if h["_step"] <= a.at]
            if not early:
                skipped += 1
                continue
            at40 = max(early)
            last_step = max(h["_step"] for h in pts)
            if at40 <= 0:
                skipped += 1
                continue
            rows.append(((ds, r.name), at40, float(end), float(end) / at40,
                         not bool(weak), last_step))

    print(f"paired within-run: {len(rows)}   skipped (missing step-40 or baseline field): {skipped}\n")
    print(f"{'group':<8}{'n':>4}{'@40':>9}{'@end':>9}{'gain':>9}{'median':>9}{'n improved':>12}")
    for lab, want in (("weak", False), ("strong", True), ("ALL", None)):
        rs = [r for r in rows if want is None or r[4] is want]
        if not rs:
            continue
        imp = sum(1 for r in rs if r[3] > 1.0001)
        print(f"{lab:<8}{len(rs):>4}{g([r[1] for r in rs]):>9.3f}{g([r[2] for r in rs]):>9.3f}"
              f"{g([r[3] for r in rs]) - 1:>8.1%}{statistics.median(r[3] for r in rs) - 1:>8.1%}"
              f"{imp:>7}/{len(rs)}")

    print(f"\n{'kernel':<52}{'@40':>8}{'@end':>8}{'gain':>8}{'steps':>7}  baseline")
    for k, s40, end, ratio, strong, steps in sorted(rows, key=lambda r: -r[3]):
        print(f"{k[0]+'/'+k[1]:<52}{s40:>8.3f}{end:>8.3f}{ratio-1:>7.0%}"
              f"{(steps if steps is not None else -1):>7}  {'strong' if strong else 'weak'}")


if __name__ == "__main__":
    main()
