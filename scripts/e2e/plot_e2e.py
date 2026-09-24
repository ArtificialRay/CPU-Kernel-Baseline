#!/usr/bin/env python3
"""Figures for the end-to-end (Qwen3.5-4B on Graviton4) experiments.

Every number here is measured, with its provenance in the DATA block below. The
default source is the definitive run (measure5.sh -> measure5c, 2026-09-24, one
c8g.4xlarge, one kernel set, results in ~/e2e/definitive/). Where a figure needs a
number that run did not produce -- the quantization ladder, the failed gate-v1 set,
the Sol arm -- the source is named on the constant.

Writes PDF (for LaTeX) and PNG (for viewing) per figure, plus a combined sheet.

  python scripts/e2e/plot_e2e.py --out sweep_backups/figs
"""
import argparse, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- measured data
# definitive run, table_20260924T1457Z.txt, pp=512 tg=128 reps=5
THREADS = [1, 4, 16]
TPUT = {  # build -> (prefill tok/s, decode tok/s) per thread count
    "stock":        [(13.7, 5.38), (53.3, 17.91), (158.2, 39.07)],
    "norepack":     [(6.7, 4.02),  (26.3, 13.21), (86.1, 33.93)],
    "agent":        [(19.3, 5.81), (74.3, 19.31), (210.3, 42.10)],
}
# definitive run, "=== pp ===" sweep, -fa off, r=3
PP = [(512, 173.55, 238.90), (2048, 165.92, 224.05),
      (4096, 155.43, 205.23), (8192, 138.50, 177.08)]
# definitive run, "=== ubatch ===" sweep, -fa off, r=3
UB = [(1, 43.41, 47.13), (2, 57.12, 82.16), (8, 127.37, 170.58),
      (64, 182.30, 198.63), (512, 173.80, 238.36)]
# 64-chunk wikitext-2 ladder: ppl64.sh (stock models) + definitive run (+kernels)
LADDER = [("Q3_K_M", 9.8459, "quant"), ("Q4_K_M\n+ agent kernels", 9.5617, "kernels"),
          ("Q4_K_M", 9.5029, "quant"), ("Q6_K", 9.4364, "quant"), ("BF16", 9.3841, "quant")]
# kernel-level geomean (gate v2, 11 kernels) vs deployed prefill speedup @t=16, and the
# perplexity each arm costs vs its own stock. Sol's ppl delta is the 8-chunk measurement
# (stock 10.062 -> 10.064); Fable's is 64-chunk (9.503 -> 9.562). Deltas are comparable,
# absolute perplexities are not.
# name, kernel-level geomean (None = not an agent), deployed prefill, ppl delta vs stock
MODELS = [("stock\nllama.cpp", None, 1.00, 0.0),
          ("gpt-5.6-sol", 1.201, 0.59, 0.0),
          ("Claude\nFable 5.1", 2.932, 1.33, 0.6)]
# gate-v1 attribution, 8 chunks, one kernel overridden at a time (docs/e2e_qwen35.md)
ATTRIB = [("ffn gate/up\nq4_K", 11.51), ("GDN in-proj\nq5_K", 11.23),
          ("GDN z-gate\nq4_K", 10.87), ("attn q\nq4_K", 10.23),
          ("lm_head\nq6_K", 10.09), ("all others\n(≤)", 10.14)]
GATE1_STOCK, GATE1_ALL = 10.062, 14.45

C_STOCK, C_AGENT, C_NOREPACK = "#6b7280", "#1f77b4", "#c7cdd6"
C_BAD, C_GOOD = "#c0392b", "#2e7d32"


def style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=10.5, pad=8, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)


def f_ladder(ax):
    names = [n for n, _, _ in LADDER]
    vals = [v for _, v, _ in LADDER]
    cols = [C_AGENT if k == "kernels" else C_STOCK for _, _, k in LADDER]
    y = range(len(names))
    ax.barh(list(y), vals, color=cols, height=0.62)
    ax.set_yticks(list(y)); ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlim(9.30, 9.92)
    for i, v in enumerate(vals):
        ax.text(v + 0.008, i, f"{v:.4f}", va="center", fontsize=8)
    style(ax, "Agent kernels cost half of one quantization step",
          "wikitext-2 perplexity (64 chunks, lower is better)")
    ax.annotate("", xy=(9.5029, 2.42), xytext=(9.3841, 2.42),
                arrowprops=dict(arrowstyle="<->", color=C_BAD, lw=1.1))
    ax.text(9.443, 2.60, "quantization  +1.27%", color=C_BAD, fontsize=8, ha="center")
    ax.annotate("", xy=(9.5617, 1.42), xytext=(9.5029, 1.42),
                arrowprops=dict(arrowstyle="<->", color=C_AGENT, lw=1.1))
    ax.text(9.532, 1.60, "kernels  +0.62%", color=C_AGENT, fontsize=8, ha="center")


def f_pp(ax):
    xs = [p[0] for p in PP]
    sp = [a / s for _, s, a in PP]
    ax.plot(range(len(xs)), sp, "o-", color=C_AGENT, lw=2, ms=6)
    for i, v in enumerate(sp):
        ax.annotate(f"{v:.3f}x", (i, v), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8.5)
    ax.axhline(1.0, color=C_STOCK, lw=1, ls="--")
    ax.text(0.02, 1.004, "stock llama.cpp", fontsize=8, color=C_STOCK)
    ax.set_xticks(range(len(xs))); ax.set_xticklabels([str(x) for x in xs])
    ax.set_ylim(0.98, 1.44)
    style(ax, "Prefill gain decays with prompt length", "prompt tokens", "speedup vs stock")


def f_tput(ax):
    w = 0.26
    for j, (lab, col) in enumerate((("norepack", C_NOREPACK), ("stock", C_STOCK), ("agent", C_AGENT))):
        vals = [v[0] for v in TPUT[lab]]
        ax.bar([i + (j - 1) * w for i in range(len(THREADS))], vals, w, label=lab, color=col)
    for i in range(len(THREADS)):
        r = TPUT["agent"][i][0] / TPUT["stock"][i][0]
        ax.text(i + w, TPUT["agent"][i][0] + 4, f"{r:.2f}x", ha="center", fontsize=8.5,
                color=C_AGENT, fontweight="bold")
    ax.set_xticks(range(len(THREADS))); ax.set_xticklabels([f"{t} thread{'s'*(t>1)}" for t in THREADS])
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    style(ax, "Prefill throughput scales with the gain intact", "", "tokens/s (pp512)")


def f_ub(ax):
    xs = [u[0] for u in UB]
    sp = [a / s for _, s, a in UB]
    cols = [C_AGENT] * len(xs)
    ax.bar(range(len(xs)), sp, 0.6, color=cols)
    for i, v in enumerate(sp):
        ax.text(i, v + 0.012, f"{v:.3f}x", ha="center", fontsize=8.5)
    ax.axhline(1.0, color=C_STOCK, lw=1, ls="--")
    ax.set_xticks(range(len(xs))); ax.set_xticklabels([str(x) for x in xs])
    ax.set_ylim(0.9, 1.56)
    style(ax, "Micro-batch: the gain survives every batch size",
          "micro-batch size (-ub)", "speedup vs stock")


def f_models(ax):
    labs = [m[0] for m in MODELS]
    x = list(range(len(labs)))
    w = 0.34
    kl = [m[1] for m in MODELS]
    ax.bar([i - w / 2 for i in x if kl[i] is not None],
           [v for v in kl if v is not None], w,
           color=C_STOCK, label="kernel-level (geomean, 11 kernels)")
    ax.bar([i + w / 2 if kl[i] is not None else i for i in x],
           [m[2] for m in MODELS], w, color=C_AGENT,
           label="deployed (prefill, end to end)")
    for i, m in enumerate(MODELS):
        if m[1] is not None:
            ax.text(i - w / 2, m[1] + 0.07, f"{m[1]:.2f}x", ha="center", fontsize=8.5)
        dx = (w / 2) if m[1] is not None else 0.0
        bad = m[2] < 1.0
        ax.text(i + dx, m[2] + 0.07, f"{m[2]:.2f}x", ha="center", fontsize=9,
                color=C_BAD if bad else "black", fontweight="bold" if bad else "normal")
    ax.axhline(1.0, color=C_BAD, lw=1, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m[0]}\n({m[3]:+.1f}% ppl)" for m in MODELS], fontsize=8.5)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_ylim(0, 3.4)
    style(ax, "Kernel-level score does not predict deployed value", "", "speedup")


def f_gate(ax):
    names = [a[0] for a in ATTRIB]
    vals = [a[1] for a in ATTRIB]
    cols = [C_BAD if v > 10.3 else C_GOOD for v in vals]
    ax.bar(range(len(names)), vals, 0.62, color=cols)
    ax.axhline(GATE1_STOCK, color=C_STOCK, lw=1.2, ls="--")
    ax.set_xlim(-0.65, len(names) - 0.1)
    ax.text(len(names) - 0.55, GATE1_STOCK + 0.04, "stock 10.06", fontsize=8,
            color=C_STOCK, ha="right", va="bottom",
            bbox=dict(fc="white", ec="none", pad=1.5))
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=7.6)
    ax.set_ylim(9.9, 11.9)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.04, f"{v:.2f}", ha="center", fontsize=8)
    style(ax, "A standard gate passed all 11; three broke the model",
          "", "perplexity, one kernel overridden at a time")
    ax.text(2.5, 11.66, f"all 11 deployed together: {GATE1_ALL} (+44%)", fontsize=8.5, ha="center",
            color=C_BAD, fontweight="bold")


FIGS = [("ppl_ladder", f_ladder), ("prefill_decay", f_pp), ("throughput", f_tput),
        ("microbatch", f_ub), ("model_vs_deployed", f_models), ("gate", f_gate)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sweep_backups/figs")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 160})

    for name, fn in FIGS:
        fig, ax = plt.subplots(figsize=(5.4, 3.5))
        fn(ax)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(out / f"e2e_{name}.{ext}", bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(11.6, 11.2))
    for (name, fn), ax in zip(FIGS, axes.ravel()):
        fn(ax)
    fig.suptitle("End-to-end: agent kernels in llama.cpp, Qwen3.5-4B-Q4_K_M on Graviton4 (c8g.4xlarge)",
                 fontsize=12, fontweight="bold", y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    for ext in ("pdf", "png"):
        fig.savefig(out / f"e2e_all.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {len(FIGS) * 2 + 2} files to {out}")


if __name__ == "__main__":
    main()
