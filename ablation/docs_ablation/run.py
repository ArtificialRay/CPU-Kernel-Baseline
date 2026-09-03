#!/usr/bin/env python3
"""Docs-ablation driver: test_scripts/bench_fleet.py with one arm applied.

    python ablation/docs_ablation/run.py --arm {control,docs,nudge} <bench_fleet args>

Everything after --arm goes to bench_fleet.py unchanged. See README.md for
the arms and for the two seams this swaps (adapter class, prepare_session).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "test_scripts"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_fleet  # noqa: E402
from arms import ARMS, adapter_class_for, hide_remote_docs  # noqa: E402


def _fleet_option(argv: list[str], name: str) -> str | None:
    """Value of `--name X` / `--name=X` in bench_fleet's argv, or None."""
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


def build_fleet_argv(arm: str, fleet_argv: list[str]) -> list[str]:
    """Pin --harness claude-code, derive a per-arm --author, and pass the
    rest through untouched."""
    harness = _fleet_option(fleet_argv, "--harness")
    if harness is None:
        fleet_argv = ["--harness", "claude-code", *fleet_argv]
    elif harness != "claude-code":
        raise SystemExit(f"docs ablation supports --harness claude-code only (got {harness!r}): "
                         "the SKILL.md strip and the prompt nudge are claude-code specific.")
    if _fleet_option(fleet_argv, "--author") is None:
        base = bench_fleet.compute_author(
            "claude-code", _fleet_option(fleet_argv, "--model"), _fleet_option(fleet_argv, "--isa") or "sve")
        fleet_argv = [*fleet_argv, "--author", f"{base}-{arm}"]
    return fleet_argv


def install_arm(arm: str) -> None:
    """Swap the two bench_fleet seams for this arm (module attributes, looked
    up by run_fleet() at call time)."""
    bench_fleet.ClaudeCodeAdapter = adapter_class_for(arm)
    if arm != "control":
        return
    original_prepare = bench_fleet.prepare_session

    def prepare_session_without_docs(target, dataset, author, isa, **kwargs):
        prepared = original_prepare(target, dataset, author, isa, **kwargs)
        try:
            hide_remote_docs(target, kwargs.get("remote_root", "~/arm-bench"), author)
        except Exception:
            bench_fleet.stop_tunnel(prepared)   # don't leave a docs-exposed "control" server up
            raise
        return prepared

    bench_fleet.prepare_session = prepare_session_without_docs


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=ARMS)
    args, fleet_argv = p.parse_known_args(argv)
    fleet_argv = build_fleet_argv(args.arm, fleet_argv)
    isa = _fleet_option(fleet_argv, "--isa") or "sve"
    os.environ.setdefault("WANDB_GROUP", f"docs-ablation-{isa}-{time.strftime('%Y%m%d')}")
    install_arm(args.arm)
    print(f"[docs_ablation] arm={args.arm} → bench_fleet {' '.join(fleet_argv)}")
    bench_fleet.main(fleet_argv)


if __name__ == "__main__":
    main()
