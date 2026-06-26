#!/usr/bin/env python3
"""Upload the simd-loop bench-trace warehouse to the HuggingFace dataset repo.

Replaces the old (pre-PR#15, conv2d-only) remote contents with the current
simd-loop set: definitions/, workloads/, and the `reference` + `autovec`
baseline solutions, under a clean `simd-loop/` path layout.

The token is read from the HF_TOKEN environment variable. It is NEVER read
from a file or hardcoded. Run with:

    HF_TOKEN=hf_... python scripts/upload_hf_trace.py            # dry-run preview
    HF_TOKEN=hf_... python scripts/upload_hf_trace.py --apply    # perform the push

Idempotent: deletes are computed from the live remote listing each run, so
re-runs never try to delete files that are already gone, and adds overwrite
in place.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationDelete,
    HfApi,
)

REPO_ID = "arm-bench/arm-bench-trace"
REPO_TYPE = "dataset"

# Repo root = parent of this script's directory.
ROOT = Path(__file__).resolve().parent.parent
TRACE = ROOT / "bench-trace"

# Local folders to upload, each mapped to the same path on the dataset.
# These are the *only* paths that should survive on the remote (plus the
# protected files below); everything else is treated as stale old layout.
UPLOAD_DIRS = [
    "definitions/simd-loop",
    "workloads/simd-loop",
    "solutions/simd-loop/reference",
    "solutions/simd-loop/autovec",
]

# Remote files that must never be deleted.
PROTECTED = {".gitattributes", "README.md", ".gitignore"}


def gather_uploads() -> list[tuple[str, Path]]:
    """Return (path_in_repo, local_path) for every file to upload."""
    out: list[tuple[str, Path]] = []
    for rel in UPLOAD_DIRS:
        base = TRACE / rel
        if not base.is_dir():
            raise SystemExit(f"missing local folder: {base}")
        for p in sorted(base.rglob("*")):
            if p.is_file():
                path_in_repo = str(p.relative_to(TRACE))
                out.append((path_in_repo, p))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--apply",
        action="store_true",
        help="actually commit (default is a dry-run preview)",
    )
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN environment variable is not set")

    api = HfApi(token=token)
    api.create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, exist_ok=True)

    uploads = gather_uploads()
    upload_paths = {pir for pir, _ in uploads}

    remote = set(api.list_repo_files(REPO_ID, repo_type=REPO_TYPE))
    keep = upload_paths | PROTECTED
    deletes = sorted(f for f in remote if f not in keep)

    print(f"local files to upload : {len(uploads)}")
    print(f"remote files (before) : {len(remote)}")
    print(f"stale files to delete : {len(deletes)}")
    for f in deletes:
        print(f"  DEL {f}")

    if not args.apply:
        print("\ndry-run only; pass --apply to commit.")
        return 0

    ops: list = [CommitOperationAdd(path_in_repo=pir, path_or_fileobj=str(p))
                 for pir, p in uploads]
    ops += [CommitOperationDelete(path_in_repo=f) for f in deletes]

    api.create_commit(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        operations=ops,
        commit_message=(
            "Replace pre-PR#15 conv2d-only trace with current simd-loop set "
            "(definitions + workloads + reference/autovec solutions)"
        ),
    )

    after = sorted(api.list_repo_files(REPO_ID, repo_type=REPO_TYPE))
    print(f"\nremote files (after)  : {len(after)}")
    for f in after:
        print(f"  {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
