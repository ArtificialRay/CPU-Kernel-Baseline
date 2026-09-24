#!/usr/bin/env python3
"""Did the agent write DIFFERENT code for each ISA, or the same code three times?

E1 reports neon 1.209 / sve 1.193 / sve2 1.229 -- indistinguishable. That null has two
opposite readings: the agent adapted well to each target and they are genuinely equivalent
for these kernels, or the agent largely ignored the target. Performance alone cannot
separate them.

The paper's Sec 5 answer is to disassemble every final kernel and count generation-specific
instruction classes; that is not done. This is the cheap partial: W&B already stores
`best_kernel_techniques` (a regex classifier over the final kernel) for every run. If the
same definition yields different technique sets across arms, the agent demonstrably wrote
different code. If they are identical everywhere, that is evidence for the second reading.

Weaker than disassembly -- a coarse classifier can miss real differences, so IDENTICAL sets
are suggestive, not proof. DIFFERENT sets are solid evidence of adaptation.

  python scripts/e2e/techniques_by_isa.py
"""
import argparse, collections
import wandb

PROJ = "ArmBench/arm-bench-kernels-gpt5.6-luna"
DS = ["ncnn", "llama.cpp", "simd-loop"]
ARMS = {"neon": ["nanobot__gpt-5.6-luna__{ds}__neon__E1_c8g"],
        "sve": ["nanobot__gpt-5.6-luna__{ds}__sve__E1_c8g"],
        "sve2": ["nanobot__gpt-5.6-luna__{ds}__sve2__S2a",
                 "nanobot__gpt-5.6-luna__{ds}__sve2__pilot",
                 "nanobot__gpt-5.6-luna__{ds}__sve2__E1_c8g"]}


def norm(v):
    if not v:
        return frozenset()
    if isinstance(v, str):
        return frozenset(t.strip() for t in v.split(",") if t.strip())
    return frozenset(str(t).strip() for t in v if str(t).strip())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--project", default=PROJ)
    a = ap.parse_args()
    api = wandb.Api(timeout=120)
    tech = collections.defaultdict(dict)
    for arm, pats in ARMS.items():
        for ds in DS:
            for p in pats:
                for r in api.runs(a.project, filters={"group": p.format(ds=ds)}, per_page=100):
                    t = norm(r.summary.get("best_kernel_techniques"))
                    if t:
                        tech[(ds, r.name)][arm] = t

    full = {k: v for k, v in tech.items() if len(v) == 3}
    print(f"kernels with techniques in all three arms: {len(full)} (of {len(tech)} seen)\n")
    same = [k for k, v in full.items() if len(set(v.values())) == 1]
    diff = [k for k, v in full.items() if len(set(v.values())) > 1]
    print(f"IDENTICAL technique set across neon/sve/sve2 : {len(same)}")
    print(f"DIFFERENT technique set in at least one arm  : {len(diff)}\n")

    counts = collections.Counter()
    for arm in ARMS:
        for k, v in full.items():
            counts[arm] += len(v[arm])
    print("mean techniques per kernel by arm:")
    for arm in ARMS:
        print(f"  {arm:<5} {counts[arm] / max(len(full), 1):.2f}")

    isa_tags = collections.defaultdict(collections.Counter)
    for arm in ARMS:
        for k, v in full.items():
            for t in v[arm]:
                isa_tags[arm][t] += 1
    tags = sorted({t for arm in isa_tags for t in isa_tags[arm]})
    print(f"\n{'technique':<26}{'neon':>7}{'sve':>7}{'sve2':>7}")
    for t in tags:
        print(f"  {t:<24}" + "".join(f"{isa_tags[arm][t]:>7}" for arm in ARMS))

    if diff:
        print(f"\nkernels where the arms differ (first 12):")
        for k in sorted(diff)[:12]:
            v = full[k]
            print(f"  {k[0]+'/'+k[1]}")
            for arm in ARMS:
                print(f"      {arm:<5} {sorted(v[arm])}")


if __name__ == "__main__":
    main()
