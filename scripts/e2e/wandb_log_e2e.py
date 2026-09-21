#!/usr/bin/env python3
"""Log one end-to-end measurement (measure_e2e.py JSON) to W&B so it lives next to the
per-kernel runs: config (model, builds, manifest kernels), the summary medians, every raw
llama-bench sample as a table, the perplexity results, and agent/stock ratios as scalars.

  python3 scripts/e2e/wandb_log_e2e.py e2e_results.json --project arm-bench-kernels-gpt5.6-luna \
      --entity ArmBench --group e2e__qwen3.5-4b --name e2e_qwen3.5-4b_fable_c8g4xl [--manifest manifest.json] [--tag ...]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--project", required=True)
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--group", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--tag", action="append", default=[])
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    import wandb

    d = json.loads(args.results.read_text())
    manifest = json.loads(args.manifest.read_text()) if args.manifest and args.manifest.exists() else None
    config = {
        "model": d["model"], "pp": d["pp"], "tg": d["tg"], "reps": d["reps"], "host": d.get("host"),
        "builds": d["builds"],
        "kernels": [{k: v for k, v in kk.items() if k != "so"} for kk in manifest["kernels"]] if manifest else None,
        "isa": manifest.get("isa") if manifest else None, "march": manifest.get("march") if manifest else None,
    }
    run = wandb.init(project=args.project, entity=args.entity, group=args.group, name=args.name,
                     job_type="e2e-measurement", tags=["e2e", *args.tag], notes=args.notes, config=config, reinit=True)

    summary = {}
    for key, v in d["summary"].items():
        build, t, kind = key.split("/")
        summary[f"median/{build}/{t}/{kind}"] = v["median"]
        summary[f"min/{build}/{t}/{kind}"] = v["min"]
        summary[f"max/{build}/{t}/{kind}"] = v["max"]
        if "speedup_vs_stock_median" in v:
            summary[f"vs_stock/{build}/{t}/{kind}"] = v["speedup_vs_stock_median"]
        base = d["summary"].get(f"norepack/{t}/{kind}")
        if base and build not in ("stock", "norepack"):
            summary[f"vs_norepack/{build}/{t}/{kind}"] = v["median"] / base["median"]
    for name, text in (d.get("perplexity") or {}).items():
        summary[f"perplexity_text/{name}"] = text
        # llama-perplexity prints e.g. "Final estimate: PPL = 8.1234 +/- 0.05"
        import re
        m = re.search(r"PPL\s*=\s*([\d.]+)(?:\s*\+/-\s*([\d.]+))?", text)
        if m:
            summary[f"perplexity/{name}"] = float(m.group(1))
            if m.group(2):
                summary[f"perplexity_err/{name}"] = float(m.group(2))
    run.summary.update(summary)

    cols = ["build", "threads", "rep", "kind", "n", "tok_per_s"]
    table = wandb.Table(columns=cols, data=[[s[c] for c in cols] for s in d["samples"]])
    run.log({"samples": table})
    med_rows = [[k.split("/")[0], int(k.split("/")[1][1:]), k.split("/")[2], v["median"], v["min"], v["max"], v.get("speedup_vs_stock_median")]
                for k, v in d["summary"].items()]
    run.log({"medians": wandb.Table(columns=["build", "threads", "kind", "median", "min", "max", "vs_stock"], data=med_rows)})
    art = wandb.Artifact(f"{args.name}-results", type="e2e-results")
    art.add_file(str(args.results))
    if manifest:
        art.add_file(str(args.manifest))
    run.log_artifact(art)
    print(f"logged {len(d['samples'])} samples, {len(d['summary'])} medians, perplexity={list((d.get('perplexity') or {}).keys())} -> {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
