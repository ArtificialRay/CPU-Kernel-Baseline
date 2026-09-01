#!/usr/bin/env python3
"""bench_fleet.py — one parametrized entry point for driving a batch
kernel-optimization run against any of this repo's harnesses

Usage:
    python3 test_scripts/bench_fleet.py --harness claude-code \\
        --dataset ncnn --isa sve2 --model anthropic/claude-opus-4-8

    python3 test_scripts/bench_fleet.py --harness nanobot \\
        --dataset ncnn --isa sve --definitions "conv2d_fp32_kh3_kw3_sh1_sw1_dh1_dw1_p1"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Reaching into skills/launch/launch_session.py's private helpers
# (_provision, _label_for) as well as its public ones is deliberate here —
# this script's whole point is to share that module's provisioning/naming
# logic instead of re-deriving it, the same way eval/mcp_client.py already
# imports its public functions.
import skills.launch.launch_session as launch_session
from skills.launch.launch_session import RemoteTarget, prepare_session, stop_tunnel, sync_results
from contracts import BASELINE_AUTHORS, ISA_INSTANCE_MAP

# harness_adapters.py lives alongside this script — Python puts a directly
# run script's own directory on sys.path[0] automatically (same idiom
# skills/launch/launch_session.py uses for its sibling remote.py).
from harness_adapters import HarnessAdapter, ClaudeCodeAdapter, NanobotAdapter, OwnHarnessAdapter, Job

DEFINITIONS_DIR = REPO_ROOT / "bench-trace" / "definitions"
EVAL_CONFIG_PATH = REPO_ROOT / "eval" / "eval_config.json"


def _free_local_port() -> int:
    """Pick an ephemeral free port for the SSH -L tunnel's local side —
    same idiom as eval/mcp_client.py::_free_local_port(). Every invocation
    of this script gets its own port so concurrent runs (different
    harness/model/isa, same --dataset) never fight over one local bind."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _sanitize_segment(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9.-]", "-", s).strip("-.")


def compute_author(harness: str, model: Optional[str], isa: str) -> str:
    """author = f"{harness}[-{model}]-{isa}". isa is always folded in —
    mcp_app/agent_tools/{ncnn,simd_loop,llama_cpp}.py names every persisted
    Solution f"{author}_{definition.name}", globally unique by
    author+definition ALONE (no isa field exists anywhere on the Solution
    model or its storage path), so two isa's sharing an author would
    otherwise silently overwrite each other's solution files. model is
    folded in whenever given, regardless of harness, so two models sharing
    a harness+isa don't collide either.
    """
    parts = [_sanitize_segment(harness)]
    if model:
        parts.append(_sanitize_segment(model.split("/")[-1]))
    parts.append(_sanitize_segment(isa))
    return "-".join(p for p in parts if p)


def build_jobs(
    dataset: str, isa: str, definitions_filter: str,
    min_iterations: int, max_iterations: int, prompt_template: str, template_args: int,
) -> list[Job]:
    """Scan bench-trace/definitions/ for every definition whose
    baseline-solution dataset (or, for simd-loop definitions, whose
    "simd-loop" tag) matches `dataset`, narrowed to `definitions_filter` if
    non-empty. `definitions_filter` is the given list for kernel definitions to run in this job
    `template_args` is 5 (nanobot's PROMPT_TEMPLATE, no max_iterations) or
    6 (claude-code's TASK_TEMPLATE, with it) — the two templates format
    the same leading fields, just claude-code's also caps iterations.
    """
    definitions_filter = definitions_filter.strip()
    if definitions_filter.startswith("["):
        raw_entries = json.loads(definitions_filter)
    else:
        raw_entries = definitions_filter.split()
    log_stem_prefix = f"{dataset}_{isa}_"
    wanted = {
        entry[len(log_stem_prefix):] if entry.startswith(log_stem_prefix) else entry
        for entry in raw_entries
    }

    jobs: list[Job] = []
    for path in sorted(DEFINITIONS_DIR.rglob("*.json")):
        d = json.loads(path.read_text())
        tags = d.get("tags", [])
        ds = next((t.split(":", 1)[1] for t in tags if t.startswith("baseline-solution:")), None)
        if ds is None and "simd-loop" in tags:
            ds = "simd-loop"
        if ds != dataset:
            continue
        name = d["name"]
        if wanted and name not in wanted:
            continue
        baseline_author = BASELINE_AUTHORS.get(ds, ds)
        args = (name, ds, baseline_author, isa, min_iterations, max_iterations)
        prompt = prompt_template % args[:template_args]
        jobs.append(Job(name=name, prompt=prompt))

    if wanted and len(jobs) != len(wanted):
        print(
            f"WARNING: requested {len(wanted)} definition(s) via --definitions, "
            f"but only found {len(jobs)} matching --dataset {dataset}.",
            file=sys.stderr,
        )
    return jobs


def _trajectory_complete(local_results_dir: Path, job_name: str) -> bool:
    """A job's local trajectory.jsonl exists and has a "submit" turn — the
    terminal marker TrajectoryWriter always emits when submit() is called
    (mcp_app/agent_tools/base.py), on both the PASSED and COMPILE_ERROR
    paths. Used to gate whether it's safe to close the MCP server/tunnel —
    if this is False for any job, its results may not have made it to
    local disk yet (or the job never got far enough to submit)."""
    traj_path = local_results_dir / job_name / "trajectory.jsonl"
    if not traj_path.exists():
        return False
    try:
        with traj_path.open() as fh:
            return any(json.loads(line).get("tool") == "submit" for line in fh if line.strip())
    except (OSError, json.JSONDecodeError):
        return False


def run_fleet(args: argparse.Namespace) -> None:
    isa, dataset = args.isa, args.dataset
    author = args.author or compute_author(args.harness, args.model, isa)
    label = args.label or launch_session._label_for(dataset, author)
    max_iterations = args.max_iterations or (args.min_iterations + 10)

    instance_type = args.instance or ISA_INSTANCE_MAP.get(isa, "c7g.large")
    instance = launch_session._provision(
        isa, instance_type, dataset, label=label, on_demand=args.on_demand,
    )
    local_port = _free_local_port()

    prepared = prepare_session(
        instance.target, dataset, author, isa,
        remote_root=args.remote_root, sync_repo=True,
        local_repo_dir=str(REPO_ROOT), local_port=local_port,
        remote_port=args.remote_port,
    )
    ran_jobs: list[Job] = []
    should_stop_tunnel = True
    try:
        adapter: HarnessAdapter
        if args.harness == "claude-code":
            adapter = ClaudeCodeAdapter(model=args.model, max_budget_usd=args.max_budget_usd,
                                        doc_nudge=args.doc_nudge)
        elif args.harness == "nanobot":
            adapter = NanobotAdapter(dataset=dataset, model=args.model, local_port=local_port)
        elif args.harness == "own":
            adapter = OwnHarnessAdapter(
                endpoint=prepared["endpoint"], author=author, remote_root=args.remote_root,
                target=instance.target, dataset=dataset, isa=isa, model=args.model,
                max_turns=max_iterations,
            )
        else:
            raise ValueError(f"Unknown --harness {args.harness!r}")

        jobs = build_jobs(
            dataset, isa, args.definitions, args.min_iterations, max_iterations,
            adapter.prompt_template, adapter.template_args,
        )
        if not jobs:
            print(
                f"No definitions found for --dataset {dataset} "
                f"(--definitions={args.definitions or '<all>'}) under {DEFINITIONS_DIR}",
                file=sys.stderr,
            )
            sys.exit(1)

        log_dir = REPO_ROOT / "harness_trajs" / args.harness / author
        log_dir.mkdir(parents=True, exist_ok=True)
        local_results_dir = Path(args.local_results_dir or (REPO_ROOT / f"agent-runs-{author}"))
        local_results_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"Running {len(jobs)} job(s) for --harness {args.harness}, --dataset {dataset}, "
            f"--isa {isa}, author={author}, label={label}"
        )

        for job in jobs:
            ran_jobs.append(job)
            log_path = log_dir / f"{dataset}_{isa}_{job.name}.log"
            attempt = 0
            while True:
                print(f"=== [{time.strftime('%H:%M:%S')}] starting job: {job.name} "
                      f"(attempt {attempt + 1}/{args.retries + 1}) ===")
                rc = adapter.run_job(job, endpoint=prepared["endpoint"], author=author, log_path=log_path)
                if rc == 0:
                    break
                if adapter.is_benign_failure(log_path):
                    print(f"  WARNING: job {job.name} crashed during MCP cleanup after "
                          f"finishing (known benign) — result may already be persisted "
                          f"remotely, continuing", file=sys.stderr)
                    break
                if attempt >= args.retries:
                    print(f"  ERROR: job {job.name}'s process exited {rc} after "
                          f"{attempt + 1} attempt(s) — giving up, see {log_path}", file=sys.stderr)
                    break
                print(f"  WARNING: job {job.name}'s process exited {rc} on attempt "
                      f"{attempt + 1} — retrying ({args.retries - attempt} retries left)",
                      file=sys.stderr)
                log_path.rename(log_path.with_suffix(log_path.suffix + f".attempt{attempt + 1}"))
                attempt += 1
            print(f"=== [{time.strftime('%H:%M:%S')}] job {job.name} finished -> {log_path} ===")
            sync_job_results(label, author, job.name, local_results_dir)
            _wandb_log_job(job.name, dataset, isa, args, author, log_path, local_results_dir)

        print(f"All jobs done. Logs in {log_dir}")
        if hasattr(adapter, "cleanup"):
            adapter.cleanup()

        if args.sync_solutions:
            print(f"Syncing bench-trace/solutions/ back from {instance.target.host}...")
            try:
                instance.target.rsync_from(
                    f"{args.remote_root}/bench-trace/solutions/", REPO_ROOT / "bench-trace" / "solutions",
                )
            except Exception as e:  # noqa: BLE001 — best-effort, never abort the batch
                print(f"  WARNING: sync-solutions failed: {e}", file=sys.stderr)

        incomplete = [j.name for j in jobs if not _trajectory_complete(local_results_dir, j.name)]
        if incomplete:
            should_stop_tunnel = False
            print(
                f"WARNING: no confirmed-complete local trajectory (no 'submit' turn found) "
                f"for: {incomplete} — leaving the MCP server/SSH tunnel running so results "
                f"aren't lost. Re-run sync-results for these once ready, then stop the tunnel "
                f"manually.", file=sys.stderr,
            )
    finally:
        for job in ran_jobs:
            adapter.cleanup_workspace(job)
        if should_stop_tunnel:
            stop_tunnel(prepared)


def _wandb_log_job(name, dataset, isa, args, author, log_path, local_results_dir) -> None:
    """Optional Weights & Biases logging (off unless WANDB=1). Parses the just-
    synced trajectory + this job's session log into one W&B run per definition
    via analysis/wandb_log_run.py. Harness-agnostic: nanobot/own-harness runs
    simply get no cost/token fields (the logger degrades gracefully). Uses
    WANDB_PYTHON (default: this interpreter), which needs `pip install wandb`;
    if it's missing the logger no-ops. Never aborts the batch."""
    if os.environ.get("WANDB", "0") != "1":
        return
    py = os.environ.get("WANDB_PYTHON", sys.executable)
    group = os.environ.get("WANDB_GROUP") or f"{args.harness}-{isa}-{time.strftime('%Y%m%d')}"
    cmd = [
        py, str(REPO_ROOT / "analysis" / "wandb_log_run.py"),
        "--name", name, "--dataset", dataset, "--isa", isa,
        "--model", args.model or "unknown", "--author", author,
        "--log-file", str(log_path), "--results-dir", str(local_results_dir),
        "--project", os.environ.get("WANDB_PROJECT", "arm-bench-kernels"),
        "--group", group,
    ]
    entity = os.environ.get("WANDB_ENTITY")
    if entity:
        cmd += ["--entity", entity]
    try:
        subprocess.run(cmd, check=False)
    except Exception as e:  # noqa: BLE001 — best-effort, never abort the batch
        print(f"  WARNING: wandb logging failed for {name} (non-fatal): {e}", file=sys.stderr)


def sync_job_results(label: str, author: str, definition: str, local_results_dir: Path) -> None:
    """Best-effort: a sync failure never aborts the rest of the batch."""
    if not EVAL_CONFIG_PATH.exists():
        print(f"  WARNING: {EVAL_CONFIG_PATH} not found — skipping sync-results for "
              f"{definition}", file=sys.stderr)
        return
    config = json.loads(EVAL_CONFIG_PATH.read_text())
    inst = config.get("instances", {}).get(label, {})
    host = inst.get("host", "")
    if not host:
        print(f"  WARNING: no host recorded for label={label} in {EVAL_CONFIG_PATH} — "
              f"skipping sync-results for {definition}", file=sys.stderr)
        return
    try:
        sync_results(
            RemoteTarget(host=host, user=inst.get("user", "ubuntu"),
                         key_file=inst.get("key_file", "~/.ssh/id_rsa")),
            author, definition=definition, local_results_dir=local_results_dir,
        )
    except Exception as e:  # noqa: BLE001 — best-effort, never abort the batch
        print(f"  WARNING: sync-results failed for {definition}: {e}", file=sys.stderr)


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--harness", required=True, choices=["claude-code", "nanobot", "own"])
    p.add_argument("--dataset", required=True, choices=["ncnn", "simd-loop", "llama.cpp"])
    p.add_argument("--isa", default="sve", choices=["neon", "sve", "sve2", "sme2"])
    p.add_argument("--model", default=None,
                   help="model override. claude-code: passed as --model to the CLI "
                        "(empty = CLI default). nanobot: patched into a temp copy of "
                        "agents.defaults.model (nanobot's CLI has no --model flag). own: "
                        "litellm model string, required (e.g. anthropic/claude-opus-4-8).")
    p.add_argument("--min-iterations", type=int, default=40,
                   help="Floor, not a cap — the model is told not to submit early.")
    p.add_argument("--max-iterations", type=int, default=None,
                   help="Soft ceiling (claude-code only). Default: --min-iterations + 10.")
    p.add_argument("--definitions", default="",
                   help="JSON array or space-separated definition names to narrow the run "
                        "to. Empty = every definition matching --dataset.")
    p.add_argument("--retries", type=int, default=3,
                   help="Retries for transient infra failures. Total attempts = retries+1.")
    p.add_argument("--author", default=None, help="Override the computed author (advanced).")
    p.add_argument("--label", default=None, help="Override the computed instance label (advanced).")
    p.add_argument("--instance", default=None, help="EC2 instance type override.")
    p.add_argument("--remote-root", default="~/arm-bench")
    p.add_argument("--remote-port", type=int, default=8765)
    p.add_argument("--on-demand", action="store_true",
                   help="Provision on-demand instead of spot (only affects a freshly "
                        "provisioned instance).")
    p.add_argument("--local-results-dir", default=None,
                   help="Default: agent-runs-<author>/ under the repo root.")
    p.add_argument("--max-budget-usd", default=None, help="claude-code only: hard $ ceiling per job.")
    p.add_argument("--doc-nudge", action="store_true",
                   help="claude-code only: append a strong nudge to read the Arm Software "
                        "Optimization Guide (MCP resource) before optimizing. OFF by default; "
                        "the treatment arm of the docs-vs-no-docs ablation.")
    p.add_argument("--sync-solutions", action="store_true",
                   help="After all jobs finish, also pull bench-trace/solutions/ back from the "
                        "remote instance (not bench-trace/traces/ — that data's already in "
                        "agent-runs-<author>/). Off by default; every harness writes solutions "
                        "to the remote bench-trace regardless of this flag.")
    args = p.parse_args(argv)
    run_fleet(args)


if __name__ == "__main__":
    main()
