#!/usr/bin/env python3
"""Additive upload of selected bench-trace files to the shared HF dataset.

arm-bench/arm-bench-trace is shared. A full-folder sync once deleted a teammate's
traces, so this script only ever uploads an explicit allow-list and never passes
delete_patterns. It refuses to run without --paths or --paths-file.

    python scripts/e2e/push_bench_trace_hf.py --paths-file paths.txt --dry-run
    python scripts/e2e/push_bench_trace_hf.py --paths-file paths.txt --yes
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO / "bench-trace"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--repo", default="arm-bench/arm-bench-trace")
    ap.add_argument("--repo-type", default="dataset")
    ap.add_argument("--paths", nargs="*", default=[], help="paths relative to --root")
    ap.add_argument("--paths-file", type=Path, default=None, help="one relative path per line")
    ap.add_argument("--message", default="additive upload from push_bench_trace_hf.py")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true", help="actually upload")
    args = ap.parse_args()

    rels = list(args.paths)
    if args.paths_file:
        rels += [l.strip() for l in args.paths_file.read_text().splitlines()
                 if l.strip() and not l.startswith("#")]
    if not rels:
        print("refusing to run with an empty allow-list (pass --paths or --paths-file)")
        return 1

    missing = [r for r in rels if not (args.root / r).exists()]
    if missing:
        print("these paths do not exist under the root:")
        for m in missing:
            print("  ", m)
        return 1

    total = 0
    for r in rels:
        p = args.root / r
        n = 1 if p.is_file() else sum(1 for _ in p.rglob("*") if _.is_file())
        total += n
        print(f"  {r}  ({n} file{'s' if n != 1 else ''})")
    print(f"\n{len(rels)} allow-list entries, {total} files -> {args.repo} ({args.repo_type})")
    print("delete_patterns is never set: nothing already in the repo can be removed by this.")

    if args.dry_run or not args.yes:
        print("\ndry run (pass --yes to upload)")
        return 0

    from huggingface_hub import HfApi
    api = HfApi()
    patterns = []
    for r in rels:
        p = args.root / r
        patterns.append(r if p.is_file() else f"{r.rstrip('/')}/**")
    api.upload_folder(
        folder_path=str(args.root),
        repo_id=args.repo,
        repo_type=args.repo_type,
        allow_patterns=patterns,
        commit_message=args.message,
    )
    print("uploaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
