"""End-to-end verification against real hardware — no nanobot needed.

Covers the two definitions confirmed (at plan time) to have baseline
solutions present in bench-trace/: conv2d_fp32_kh1_kw1_sh1_sw1_dh1_dw1_p0
(ncnn) and gemm_bf16_n1024_k2048 (llama.cpp). Requires a provisioned
sve2-tier instance beforehand — by whatever means (e.g. `python
eval/provision.py --isa sve2`, run separately; this script never imports
eval.provision). See mcp_app/README.md, Verification Step 1.

Usage:
    python -m mcp_app.scripts.smoke_test --host 1.2.3.4 --user ubuntu --key-file ~/.ssh/id_rsa
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from . import _local_ssh
from .test_mcp_client import run_stdio_sequence

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_RSYNC_EXCLUDES = [
    "build", ".git", "terraform", "generations", "results", "notebooks",
    "agent-runs", "agent-runs-mcp", "agent-runs-nanobot", "__pycache__", "*.pyc",
]

VERIFICATION_DEFINITIONS = [
    ("conv2d_fp32_kh1_kw1_sh1_sw1_dh1_dw1_p0", "ncnn", "baseline-ncnn-arm"),
    ("gemm_bf16_n1024_k2048", "llama.cpp", "baseline-llamacpp-arm"),
]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", required=True)
    p.add_argument("--user", default="ubuntu")
    p.add_argument("--key-file", default="~/.ssh/id_rsa")
    p.add_argument("--remote-root", default="~/arm-bench")
    p.add_argument("--isa", default="sve2")
    p.add_argument("--no-sync", action="store_true", help="Skip rsyncing the repo up first.")
    args = p.parse_args(argv)

    if not args.no_sync:
        print(f"[smoke_test] rsyncing repo to {args.user}@{args.host}:{args.remote_root} ...")
        _local_ssh.rsync_to(
            args.host, args.user, args.key_file, REPO_ROOT, args.remote_root,
            excludes=DEFAULT_RSYNC_EXCLUDES,
        )

    failures = []
    for def_name, dataset, baseline_author in VERIFICATION_DEFINITIONS:
        print(f"\n[smoke_test] {dataset}/{def_name} (isa={args.isa}) ...")
        remote_cmd = (
            f"cd {args.remote_root} && python3 -m mcp_app.server --dataset {dataset} "
            f"--definition {def_name} --author smoke-test --baseline-author {baseline_author} "
            f"--isa {args.isa} --run-dir {args.remote_root}/agent-runs-mcp/{def_name} "
            "--transport stdio"
        )
        spawn_args = _local_ssh.ssh_spawn_args(args.host, args.user, args.key_file, remote_cmd)

        try:
            result = asyncio.run(run_stdio_sequence("ssh", spawn_args, verbose=True))
            print(f"[smoke_test] {def_name}: PASSED — {result}")
        except AssertionError as e:
            print(f"[smoke_test] {def_name}: FAILED — {e}")
            failures.append(def_name)
            continue
        finally:
            _local_ssh.rsync_from(
                args.host, args.user, args.key_file,
                f"{args.remote_root}/agent-runs-mcp/{def_name}",
                REPO_ROOT / "agent-runs-nanobot",
            )

        traj = REPO_ROOT / "agent-runs-nanobot" / def_name / "trajectory.jsonl"
        if not traj.exists():
            print(f"[smoke_test] {def_name}: WARNING trajectory.jsonl not found after sync at {traj}")
            failures.append(def_name)

    if failures:
        print(f"\n[smoke_test] FAILED: {failures}")
        sys.exit(1)
    print(f"\n[smoke_test] PASSED for all {len(VERIFICATION_DEFINITIONS)} definitions")


if __name__ == "__main__":
    main()
