#!/usr/bin/env python3
"""bench_fleet.py — one parametrized entry point for driving a batch
kernel-optimization run against any of this repo's harnesses

Usage:
    python3 test_scripts/bench_fleet.py --harness claude-code \\
        --dataset ncnn --isa sve2 --model anthropic/claude-opus-4-8

    python3 test_scripts/bench_fleet.py --harness nanobot \\
        --dataset ncnn --isa sve --definitions "conv2d_fp32_kh3_kw3_sh1_sw1_dh1_dw1_p1"

    python3 test_scripts/bench_fleet.py --harness codex \\
        --dataset ncnn --isa sve2 --model gpt-5.6-luna

    python3 test_scripts/bench_fleet.py --harness cline \\
        --dataset ncnn --isa sve2 --model gpt-5.6-luna

    # Resume across multiple datasets, processing them in order and running
    # chunks for each dataset on up to two instances:
    python3 test_scripts/bench_fleet.py --harness claude-code \\
        --dataset ncnn llama.cpp --isa sve2 --model sonnet --until-complete \\
        --max-parallel-instances 2
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import analysis.wandb_log_run as wandb_log_run

# Reaching into skills/launch/launch_session.py's private helpers
# (_provision, _label_for) as well as its public ones is deliberate here —
# this script's whole point is to share that module's provisioning/naming
# logic instead of re-deriving it, the same way eval/mcp_client.py already
# imports its public functions.
import skills.launch.launch_session as launch_session
from skills.launch.launch_session import prepare_session, stop_tunnel, sync_results
from bench.config import EvalConfig
from contracts import ISA_INSTANCE_MAP, baseline_author_for
from mcp_app.agent_tools.isa import isa_satisfies

# harness_adapters.py lives alongside this script — Python puts a directly
# run script's own directory on sys.path[0] automatically (same idiom
# skills/launch/launch_session.py uses for its sibling remote.py).
from harness_adapters import (
    HarnessAdapter, HarnessLaunchError, ClaudeCodeAdapter, ClineAdapter, CodexAdapter, NanobotAdapter,
    OwnHarnessAdapter, SingleShotAdapter, Job, QUOTA_EXIT_CODE,
)

DEFINITIONS_DIR = REPO_ROOT / "bench-trace" / "definitions"

# For run_until_complete()'s round planner only — it needs each harness's
# prompt_template/template_args.
ADAPTER_CLASSES = {
    "claude-code": ClaudeCodeAdapter, "cline": ClineAdapter, "codex": CodexAdapter,
    "nanobot": NanobotAdapter, "own": OwnHarnessAdapter,
    "single-shot": SingleShotAdapter,
}


class FleetRunError(RuntimeError):
    """A batch stopped before every requested job completed."""


class QuotaExceeded(FleetRunError):
    """The harness account is out of quota; retrying jobs cannot help."""


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


def _cost_proxy(name: str) -> int:
    """Crude workload-size proxy: product of the integers in the
    definition name (pooling/1x1 convs sort first, kh7_kw7 mid, big gemms
    last), so a --until-complete resume yields fast sanity-check data
    early and defers the slowest evals to the tail."""
    prod = 1
    for x in re.findall(r"\d+", name):
        prod *= max(int(x), 1)
    return prod


def ensure_baselines(
    instance, dataset: str, definitions: list[str], remote_root: str, isa: Optional[str] = None,
) -> None:
    """Sync the repo, then collect + verify baseline traces BEFORE the MCP
    server starts. The author is ISA-routed (contracts.baseline_author_for) so
    it matches what mcp_app/session.py will compare against on this box.
    """
    baseline_author = baseline_author_for(dataset, isa)
    target = instance.target
    # Every tier's uv-managed venv (provisioning/provision.py::_venv_steps) —
    # same path as provisioning/remote.py::InstanceHandle.python, which
    # RemoteTarget can't expose directly (skills/launch/remote.py carries no
    # instance_type, and none is needed any more).
    python = "~/venv/bin/python3"

    print(f"[sync] Syncing benchmark inputs to {target.host} before baseline collection...")
    target.rsync_to(str(REPO_ROOT), remote_root, paths=launch_session.RSYNC_ALLOWLIST)

    missing = [
        d for d in definitions
        if not _has_passed_baseline(target, python, d, baseline_author, remote_root)
    ]
    if not missing:
        print(f"[baselines] All {len(definitions)} baseline trace(s) present.")
        return

    print(f"[baselines] {len(missing)}/{len(definitions)} definition(s) missing baseline "
          f"traces — collecting (author={baseline_author!r})...")
    for i, name in enumerate(missing):
        print(f"  [{i + 1}/{len(missing)}] {name} ...", end=" ", flush=True)
        rc, out, err = target.run(
            f"cd {remote_root} && {python} -m bench.cli collect-baselines "
            f"--baseline-author {baseline_author} --definition {name}",
            timeout=1500,
        )
        print("OK" if rc == 0 else "FAILED")
        if rc != 0:
            combined = "\n".join(filter(None, [out.strip(), err.strip()]))
            raise RuntimeError(
                f"Baseline collection failed for {name}; refusing to start the agent "
                f"(speedup would come back None).\n{combined}\n\n"
            )

    still_missing = [
        d for d in missing
        if not _has_passed_baseline(target, python, d, baseline_author, remote_root)
    ]
    if still_missing:
        raise RuntimeError(
            f"collect-baselines reported success but no PASSED baseline trace exists for: "
            f"{', '.join(still_missing)}"
        )


def _has_passed_baseline(
    target, python: str, definition: str, baseline_author: str, remote_root: str,
) -> bool:
    """True when `definition` already has a PASSED trace for `baseline_author`
    on this host, measured under the current timing protocol. Host-specific on
    purpose: baselines are absolute timings, so one is only valid on the
    machine that measured it. Protocol-specific for the same reason the
    server's own check is (mcp_app/agent_tools/baseline_readiness.py): a
    baseline timed another way must be re-collected, not reused."""
    check = (
        "import json,sys\n"
        "from pathlib import Path\n"
        f"td = Path({remote_root!r}).expanduser() / 'bench-trace' / 'traces'\n"
        f"for f in td.rglob({definition!r} + '.jsonl'):\n"
        "    for line in f.open():\n"
        "        line = line.strip()\n"
        "        if not line:\n"
        "            continue\n"
        "        r = json.loads(line)\n"
        f"        sol = r.get('solution','')\n"
        f"        ev = r.get('evaluation') or {{}}\n"
        f"        perf = ev.get('performance') or {{}}\n"
        f"        if (sol.startswith({baseline_author!r}) and ev.get('status') == 'PASSED'\n"
        f"                and perf.get('timing_protocol') == {EvalConfig().timing_protocol!r}):\n"
        "            sys.exit(0)\n"
        "sys.exit(1)\n"
    )
    b64 = base64.b64encode(check.encode()).decode()
    rc, _, _ = target.run(f"echo {b64!r} | base64 -d | {python}", timeout=60)
    return rc == 0


def build_jobs(
    dataset: str, isa: str, definitions_filter: str,
    min_iterations: int, max_iterations: int, prompt_template: str, template_args: int,
) -> list[Job]:
    """Build one Job (with its rendered prompt) per definition matching
    `dataset`, narrowed to `definitions_filter` if non-empty (a JSON array
    or space-separated list of names). Definitions tagged `e2e:<model>`
    (k-quant gemms taken from a model's matmul shapes) are run only when
    named in `definitions_filter`, never as part of a whole-dataset sweep.
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

    baseline_author = baseline_author_for(dataset, isa)
    jobs: list[Job] = []
    isa_skipped: list[str] = []
    e2e_skipped: list[str] = []
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
        if not wanted and any(t.startswith("e2e:") for t in tags):
            e2e_skipped.append(name)
            continue
        if not _baseline_isa_compatible(dataset, baseline_author, d["op_type"], name, isa):
            isa_skipped.append(name)
            continue
        args = (name, dataset, baseline_author, isa, min_iterations, max_iterations)
        prompt = prompt_template % args[:template_args]
        jobs.append(Job(name=name, prompt=prompt, op_type=d["op_type"]))

    if e2e_skipped:
        print(
            f"[e2e-filter] skipping {len(e2e_skipped)} e2e definition(s); name them in "
            f"--definitions to run them: {', '.join(e2e_skipped)}",
            file=sys.stderr,
        )
    if isa_skipped:
        print(
            f"[isa-filter] skipping {len(isa_skipped)} definition(s) whose baseline solution's "
            f"isa_features aren't satisfied by --isa {isa}: {', '.join(isa_skipped)}",
            file=sys.stderr,
        )
    if wanted and len(jobs) + len(isa_skipped) != len(wanted):
        print(
            f"WARNING: requested {len(wanted)} definition(s) via --definitions, "
            f"but only found {len(jobs) + len(isa_skipped)} matching --dataset {dataset}.",
            file=sys.stderr,
        )
    return jobs


def _baseline_isa_compatible(
    dataset: str, baseline_author: str, op_type: str, name: str, isa: str,
) -> bool:
    """True unless this definition's local baseline solution declares
    isa_features the current --isa can't satisfy 、
    """
    sol_path = (
        REPO_ROOT / "bench-trace" / "solutions" / dataset / baseline_author / op_type / f"{name}.json"
    )
    if not sol_path.exists():
        return True
    isa_features = json.loads(sol_path.read_text()).get("spec", {}).get("isa_features", [])
    if not isa_features:
        return True
    return isa_satisfies(isa_features, isa)


def _trajectory_complete(local_results_dir: Path, job_name: str, min_iterations: int) -> bool:
    """A job's local trajectory.jsonl exists, has a "submit" turn — the
    terminal marker TrajectoryWriter always emits when submit() is called
    (mcp_app/agent_tools/base.py), on both the PASSED and COMPILE_ERROR
    paths — AND has at least `min_iterations` non-submit tool calls recorded 
    before it. """
    traj_path = local_results_dir / job_name / "trajectory.jsonl"
    if not traj_path.exists():
        return False
    try:
        tools = [json.loads(line).get("tool") for line in traj_path.open() if line.strip()]
    except (OSError, json.JSONDecodeError):
        return False
    if "submit" not in tools:
        return False
    exploration_calls = sum(1 for t in tools if t != "submit")
    return exploration_calls >= min_iterations


def run_fleet(args: argparse.Namespace, dataset: str) -> str:
    """Provision/reuse one instance, drive `dataset`'s matching definitions
    (narrowed by --definitions) through the chosen harness exactly once,
    sync, and close the session. This is the single-pass primitive —
    --until-complete's run_until_complete() re-invokes this script as a
    fresh subprocess per (dataset, chunk) instead of calling this directly,
    so each round/chunk gets its own process (see _run_chunk_subprocess's
    docstring for why). 
    
    Returns the instance label this call provisioned/
    reused, so main() can scope its final teardown to just this instance
    instead of tearing down every instance provisioning/provision.py knows about."""
    adapter: HarnessAdapter
    model = args.model
    local_port = _free_local_port()
    if args.harness == "claude-code":
        adapter = ClaudeCodeAdapter(model=args.model)
    elif args.harness == "codex":
        adapter = CodexAdapter(model=args.model)
    elif args.harness == "cline":
        adapter = ClineAdapter(model=args.model)
    elif args.harness == "nanobot":
        adapter = NanobotAdapter(dataset=dataset, model=args.model, local_port=local_port)
        if model is None:
            model = adapter.model
    elif args.harness not in ("own", "single-shot"):
        raise ValueError(f"Unknown --harness {args.harness!r}")
    isa = args.isa
    author = args.author or compute_author(args.harness, model, isa)
    label = args.label or launch_session._label_for(dataset, author)
    max_iterations = args.max_iterations
    instance_type = args.instance or ISA_INSTANCE_MAP.get(isa, "c7g.large")
    instance = launch_session._provision(
        isa, instance_type, dataset, label=label, on_demand=args.on_demand,
    )
    # Build the job list first: ensure_baselines() needs the definition names,
    # and it has to run between the repo sync and the MCP server start.
    # Use the class (not `adapter`) — for --harness own, adapter isn't
    # constructed until prepare_session() below hands it an MCP endpoint.
    adapter_cls = ADAPTER_CLASSES[args.harness]
    jobs = build_jobs(
        dataset, isa, args.definitions, args.min_iterations, max_iterations,
        adapter_cls.prompt_template, adapter_cls.template_args,
    )
    # Kept armed through baseline collection too: on a fresh box it can run
    # longer than one watchdog window.
    with WatchdogKeeper(instance, args.watchdog_minutes):
        if jobs:
            ensure_baselines(
                instance, dataset, [j.name for j in jobs], args.remote_root, isa=isa,
            )

    # sync_repo=False: ensure_baselines() already synced, and re-syncing here
    # would rsync --delete the baselines it just collected.
    prepared = prepare_session(
        instance.target, dataset, author, isa,
        remote_root=args.remote_root, sync_repo=False,
        local_repo_dir=str(REPO_ROOT), local_port=local_port,
        remote_port=args.remote_port, max_iterations=max_iterations,
        instance_type=instance.instance_type,
    )
    # "own" needs the just-established MCP endpoint/target, unlike
    # claude-code/nanobot above — construct it here instead.
    if args.harness == "own":
        adapter = OwnHarnessAdapter(
            endpoint=prepared["endpoint"], author=author, remote_root=args.remote_root,
            target=instance.target, dataset=dataset, isa=isa, model=args.model,
            # Deliberately decoupled from the MCP server's own --max-iterations
            # cap (line above, `max_iterations`): since some model won't provide a tool call for each turn,
            # set a tool call budget that is much higher than max-iterations to avoid unnecessary re-running
            max_turns=max_iterations * 3,
        )
    elif args.harness == "single-shot":
        adapter = SingleShotAdapter(
            endpoint=prepared["endpoint"], author=author, remote_root=args.remote_root,
            target=instance.target, dataset=dataset, isa=isa, model=args.model,
            samples=args.samples, temperature=args.temperature,
        )
    ran_jobs: list[Job] = []
    failed_jobs: list[str] = []
    try:
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
            quota_hit = False
            while True:
                print(f"=== [{time.strftime('%H:%M:%S')}] starting job: {job.name} "
                      f"(attempt {attempt + 1}/{args.retries + 1}) ===")
                try:
                    with WatchdogKeeper(instance, args.watchdog_minutes):
                        rc = adapter.run_job(
                            job, endpoint=prepared["endpoint"], author=author, log_path=log_path,
                        )
                except HarnessLaunchError as exc:
                    if attempt < args.retries:
                        print(f"  WARNING: could not start {job.name}: {exc}; retrying",
                              file=sys.stderr)
                        time.sleep(min(2 ** attempt, 10))
                        attempt += 1
                        continue
                    log_path.write_text(f"Harness process could not start: {exc}\n")
                    rc = 1
                    print(f"  ERROR: could not start {job.name} after {attempt + 1} "
                          f"attempt(s): {exc}", file=sys.stderr)
                    break
                quota_hit = adapter.quota_exceeded(log_path, rc)
                if rc != 0 and not quota_hit:
                    print(f"  ERROR: job {job.name}'s process exited {rc}; "
                          f"see {log_path}", file=sys.stderr)
                break
            print(f"=== [{time.strftime('%H:%M:%S')}] job {job.name} finished -> {log_path} ===")
            sync_job_artifacts(
                instance, args.remote_root, dataset, author, job, local_results_dir,
                sync_solutions=args.sync_solutions,
            )
            wandb_log_job(job.name, dataset, isa, args, author, local_results_dir)
            if quota_hit:
                raise QuotaExceeded(
                    f"{args.harness} account quota exhausted during {job.name}; stopped the batch "
                    f"without starting further jobs. See {log_path}"
                )
            if rc != 0:
                failed_jobs.append(job.name)

        print(f"All jobs done. Logs in {log_dir}")

        incomplete = [
            j.name for j in jobs
            if not _trajectory_complete(local_results_dir, j.name, args.min_iterations)
        ]
        if incomplete:
            print(
                f"WARNING: no confirmed-complete local trajectory (no 'submit' turn, or fewer "
                f"than --min-iterations {args.min_iterations} exploration tool calls before it) "
                f"for: {incomplete} — stopping the session anyway; a rerun resumes each from its "
                f"last checkpointed version (see mcp_app/agent_tools/trajectory.py). Results already "
                f"synced above reflect progress up to that point.", file=sys.stderr,
            )
        if failed_jobs or incomplete:
            raise FleetRunError(
                f"Batch incomplete: failed processes={failed_jobs}, "
                f"incomplete trajectories={incomplete}"
            )
    finally:
        if hasattr(adapter, "cleanup"):
            try:
                adapter.cleanup()
            except Exception as e:  # noqa: BLE001 — retain the job's original failure
                print(f"  WARNING: adapter cleanup failed: {e}", file=sys.stderr)
        for job in ran_jobs:
            adapter.cleanup_workspace(job)
        stop_tunnel(prepared)
    return label


class WatchdogKeeper:
    """Re-arm the box's watchdog on entry and then every third of the window
    while a job runs: a 40-iteration job can outlast one window, and the box
    must not shut down under it. With minutes <= 0 it cancels the pending
    shutdown instead."""

    def __init__(self, instance, minutes: int) -> None:
        self.instance, self.minutes = instance, minutes
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.keep_armed, daemon=True, name="watchdog-keeper")

    def keep_armed(self) -> None:
        period_s = max(5, self.minutes // 3) * 60
        while not self.stopped.wait(period_s):
            launch_session.arm_watchdog(self.instance, self.minutes)

    def __enter__(self) -> "WatchdogKeeper":
        launch_session.arm_watchdog(self.instance, self.minutes)
        if self.minutes > 0:
            self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stopped.set()


def _run_chunk_subprocess(
    args: argparse.Namespace, dataset: str, definitions: list[str], author: str, label: str,
) -> int:
    """One round's chunk = one fresh `bench_fleet.py` subprocess with no --until-complete.
    Deliberately call a subprocess enable each chunk gets its own process, so that what
    happened in the subprocess (e.g. a wedged SSH tunnel, a leaked adapter thread) never
    affects the next chunk."""
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--harness", args.harness, "--dataset", dataset, "--isa", args.isa,
        "--min-iterations", str(args.min_iterations),
        "--retries", str(args.retries),
        "--author", author, "--label", label,
        "--remote-root", args.remote_root, "--remote-port", str(args.remote_port),
        "--definitions", json.dumps(definitions),
        # This is a single-pass invocation of this same script (see this
        # function's own docstring) — without this flag it would hit
        # main()'s unconditional post-run teardown and destroy every
        # instance 
        "--skip-final-teardown",
    ]
    if args.model:
        cmd += ["--model", args.model]
    if args.max_iterations:
        cmd += ["--max-iterations", str(args.max_iterations)]
    if args.instance:
        cmd += ["--instance", args.instance]
    if args.on_demand:
        cmd.append("--on-demand")
    cmd += ["--watchdog-minutes", str(args.watchdog_minutes)]
    if args.local_results_dir:
        cmd += ["--local-results-dir", args.local_results_dir]
    if args.sync_solutions:
        cmd.append("--sync-solutions")
    if args.wandb:
        cmd += ["--wandb", "--wandb-project", args.wandb_project]
        if args.wandb_entity:
            cmd += ["--wandb-entity", args.wandb_entity]
        # explicit group (not just the default formula) so every chunk/round
        # in this sweep lands in the same W&B group even if a caller passed
        # a custom --wandb-group.
        cmd += ["--wandb-group", args.wandb_group or f"{args.harness}-{args.isa}-{time.strftime('%Y%m%d')}"]
    print(f"=== [{time.strftime('%H:%M:%S')}] chunk: {dataset} ({len(definitions)} def(s)) ===")
    return subprocess.run(cmd, check=False).returncode


def run_until_complete(args: argparse.Namespace) -> list[str]:
    """Resume each dataset with a bounded pool of instance workers.

    A definition keeps the same worker across rounds so its remote checkpoint
    remains available. Workers run their own chunks serially and sync results
    by definition; different workers run concurrently. Datasets use the same
    pool in sequence, keeping the instance count within the configured limit.
    """
    datasets = args.dataset
    model = args.model or ADAPTER_CLASSES[args.harness].default_model()
    author = args.author or compute_author(args.harness, model, args.isa)
    base_label = args.label or launch_session._label_for(datasets[0], author)
    worker_limit = args.max_parallel_instances
    local_results_dir = Path(args.local_results_dir or (REPO_ROOT / f"agent-runs-{author}"))
    adapter_cls = ADAPTER_CLASSES[args.harness]
    jobs_by_dataset = {
        ds: build_jobs(
            ds, args.isa, args.definitions, args.min_iterations, args.max_iterations,
            adapter_cls.prompt_template, adapter_cls.template_args,
        )
        for ds in datasets
    }
    seen_names: dict[str, str] = {}
    for ds, jobs in jobs_by_dataset.items():
        for job in jobs:
            previous = seen_names.setdefault(job.name, ds)
            if previous != ds:
                raise FleetRunError(
                    f"Definition {job.name!r} occurs in both {previous} and {ds}; "
                    "their agent-runs paths would collide"
                )

    used_labels: set[str] = set()
    for ds in datasets:
        # Partition the full job list once. Filtering to incomplete jobs before
        # assignment would move a resumed job to another instance.
        ordered = sorted(jobs_by_dataset[ds], key=lambda j: (_cost_proxy(j.name), j.name))
        worker_count = min(worker_limit, len(ordered))
        if not worker_count:
            continue
        lanes = [[job.name for job in ordered[i::worker_count]]
                 for i in range(worker_count)]
        labels = [base_label if i == 0 else f"{base_label}-worker-{i + 1}"
                  for i in range(worker_count)]
        previous_counts: list[Optional[int]] = [None] * worker_count

        for round_num in range(1, args.max_rounds + 1):
            incomplete_lanes = [
                [name for name in lane
                 if not _trajectory_complete(local_results_dir, name, args.min_iterations)]
                for lane in lanes
            ]
            remaining = sum(len(names) for names in incomplete_lanes)
            if remaining == 0:
                print(f"=== [{time.strftime('%H:%M:%S')}] {ds}: ALL COMPLETE "
                      f"at round {round_num} ===")
                break
            print(f"=== [{time.strftime('%H:%M:%S')}] {ds}: round "
                  f"{round_num}/{args.max_rounds}, {remaining} incomplete "
                  f"across {worker_count} instance slot(s) ===")

            for i, names in enumerate(incomplete_lanes):
                count = len(names)
                if count and previous_counts[i] == count:
                    label = labels[i]
                    if launch_session.is_instance_reachable(label):
                        print(f"  STALLED but {label} is reachable "
                              f"({count} incomplete); leaving it running", file=sys.stderr)
                    else:
                        print(f"  UNREACHABLE: {label} ({count} incomplete); "
                              "tearing down for reprovisioning", file=sys.stderr)
                        try:
                            launch_session._teardown(label)
                        except Exception as e:  # noqa: BLE001 — retry provisioning next round
                            print(f"  WARNING: teardown failed for {label}: {e}", file=sys.stderr)
                previous_counts[i] = count

            stop = threading.Event()

            def run_lane(i: int) -> None:
                label = labels[i]
                names = incomplete_lanes[i]
                for start in range(0, len(names), args.batch_size):
                    if stop.is_set():
                        return
                    chunk = names[start:start + args.batch_size]
                    rc = _run_chunk_subprocess(args, ds, chunk, author, label)
                    if rc == QUOTA_EXIT_CODE:
                        stop.set()
                        raise QuotaExceeded(
                            f"{ds}: harness quota exhausted; stopping new chunks"
                        )
                    if rc != 0:
                        print(f"  WARNING: {ds} chunk on {label} exited {rc}; "
                              "its incomplete jobs remain eligible next round",
                              file=sys.stderr)

            active = [i for i, names in enumerate(incomplete_lanes) if names]
            used_labels.update(labels[i] for i in active)
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(active)) as executor:
                futures = [executor.submit(run_lane, i) for i in active]
                try:
                    for future in concurrent.futures.as_completed(futures):
                        future.result()
                except Exception:
                    stop.set()
                    raise

            if all(_trajectory_complete(local_results_dir, job.name, args.min_iterations)
                   for job in ordered):
                print(f"=== [{time.strftime('%H:%M:%S')}] {ds}: ALL COMPLETE "
                      f"after round {round_num} ===")
                break
        else:
            still_incomplete = [
                job.name for job in ordered
                if not _trajectory_complete(local_results_dir, job.name, args.min_iterations)
            ]
            raise FleetRunError(
                f"{ds}: still incomplete after {args.max_rounds} rounds: {still_incomplete}"
            )
    return sorted(used_labels)


def wandb_log_job(name, dataset, isa, args, author, local_results_dir) -> None:
    """Optional Weights & Biases logging (off unless --wandb). Hands the
    just-synced trajectory to analysis/wandb_log_run.py."""
    if not args.wandb:
        return
    try:
        wandb_log_run.log_run_to_wandb(
            name=name, dataset=dataset, isa=isa, model=args.model or "unknown", author=author,
            harness=args.harness,
            trajectory_path=wandb_log_run._locate_trajectory(str(local_results_dir), name),
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group or f"{args.harness}-{isa}-{time.strftime('%Y%m%d')}",
        )
    except Exception as e:  # noqa: BLE001 — best-effort, never abort the batch
        print(f"  WARNING: wandb logging failed for {name} (non-fatal): {e}", file=sys.stderr)


def sync_job_artifacts(
    instance, remote_root: str, dataset: str, author: str, job: Job,
    local_results_dir: Path, *, sync_solutions: bool,
) -> None:
    """Pull this job's trajectory and, optionally, its best solution."""
    try:
        sync_results(
            instance.target, author, definition=job.name,
            remote_root=remote_root, local_results_dir=local_results_dir,
        )
    except Exception as e:  # noqa: BLE001 — best-effort, never abort the batch
        print(f"  WARNING: sync-results failed for {job.name}: {e}", file=sys.stderr)

    if not sync_solutions:
        return
    remote_dir = f"{remote_root}/bench-trace/solutions/{dataset}/{author}/{job.op_type}"
    local_dir = REPO_ROOT / "bench-trace" / "solutions" / dataset / author / job.op_type
    canonical_name = f"{job.name}.json"
    try:
        # submit() normally renames _best to the canonical name. If both are
        # present after an interrupted submit, _best is the newer candidate.
        for filename in (f"{job.name}_best.json", canonical_name):
            remote_file = f"{remote_dir}/{filename}"
            rc, _, _ = instance.target.run(f"test -f {remote_file}", timeout=15)
            if rc != 0:
                continue
            local_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=local_dir) as staging_dir:
                instance.target.rsync_from(remote_file, staging_dir)
                (Path(staging_dir) / filename).replace(local_dir / canonical_name)
            return
    except Exception as e:  # noqa: BLE001 — best-effort, never abort the batch
        print(f"  WARNING: sync-solutions failed for {job.name}: {e}", file=sys.stderr)


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--harness", required=True,
                   choices=["claude-code", "cline", "codex", "nanobot", "own", "single-shot"])
    p.add_argument("--dataset", required=True, nargs="+", choices=["ncnn", "simd-loop", "llama.cpp", "kleidiai"],
                   help="One or more datasets (space-separated). More than one requires "
                        "--until-complete, which processes them in order.")
    p.add_argument("--isa", default="sve", choices=["neon", "sve", "sve2", "sme2", "portable"])
    p.add_argument("--model", default=None,
                   help="model override. claude-code: passed as --model to the CLI "
                        "(empty = CLI default). nanobot: patched into a temp copy of "
                        "agents.defaults.model (nanobot's CLI has no --model flag). own: "
                        "litellm model string, required (e.g. anthropic/claude-opus-4-8).")
    p.add_argument("--samples", type=int, default=3,
                   help="--harness single-shot only: independent one-shot generations per "
                        "definition. Failed samples are kept, not redrawn — the pass rate is "
                        "part of the result. Default: 3")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="--harness single-shot only: sampling temperature. The default of 1.0 "
                        "is what makes repeated samples differ; 0 would draw the same kernel "
                        "every time and waste --samples. Default: 1.0")
    p.add_argument("--min-iterations", type=int, default=15,
                   help="Floor, not a cap — the model is told not to submit early. Default: 15")
    p.add_argument("--max-iterations", type=int, default=40,
                   help="Ceiling, baked into every harness's prompt as a soft budget and "
                        "into the MCP server as a hard --max-iterations cap (mcp_app/server.py) "
                        "that rejects further compile/evaluate/disassemble calls once hit. "
                        "Default: 40, the benchmark's 40-step budget")
    p.add_argument("--definitions", default="",
                   help="JSON array or space-separated definition names to narrow the run "
                        "to. Empty = every definition matching --dataset, except the ones "
                        "tagged e2e:<model>, which run only when named here.")
    p.add_argument("--retries", type=int, default=3,
                   help="Retries only when the OS temporarily cannot start a harness process. "
                        "Model/API errors are handled within the harness. Default: 3.")
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
    p.add_argument("--watchdog-minutes", type=int, default=120,
                   help="Cost guard for Linux boxes: the box shuts itself down (and terraform "
                        "terminates it) this many minutes after the last re-arm. Kept armed while "
                        "baselines are collected and during every job, so it only fires once the driver is gone. 0 "
                        "cancels it. Default: 120")
    p.add_argument("--sync-solutions", action="store_true",
                   help="After each job, pull only its best solution to the local canonical "
                        "<definition>.json file. Prefer a remote _best when present; ignore "
                        "_current. Off by default.")
    p.add_argument("--until-complete", action="store_true",
                   help="Resume until each definition has a confirmed-complete trajectory. "
                        "Datasets run in order; within one dataset, jobs stay assigned to "
                        "instance workers across rounds. Each chunk runs in a fresh subprocess.")
    p.add_argument("--max-rounds", type=int, default=8,
                   help="--until-complete only: give up after this many rounds.")
    p.add_argument("--batch-size", type=int, default=3,
                   help="--until-complete: jobs per chunk on each instance worker. Default: 3.")
    p.add_argument("--max-parallel-instances", type=int, default=1,
                   help="--until-complete: maximum instances managed by this command. "
                        "Default: 1; use 2 or more to run chunks in parallel.")
    p.add_argument("--wandb", action="store_true",
                   help="Log every job to Weights & Biases (needs `pip install wandb`). Off by "
                        "default.")
    p.add_argument("--wandb-project", default="arm-bench-kernels")
    p.add_argument("--wandb-entity", default=None, help="Default: your wandb default entity.")
    p.add_argument("--wandb-group", default=None,
                   help="Default: '{harness}-{isa}-{today's date}', so one sweep's jobs land in "
                        "one group.")
    p.add_argument("--skip-final-teardown", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    if (args.batch_size < 1 or args.max_rounds < 1 or args.max_parallel_instances < 1
            or args.retries < 0):
        p.error("--batch-size, --max-rounds and --max-parallel-instances must be positive; "
                "--retries must be nonnegative")
    if len(args.dataset) != len(set(args.dataset)):
        p.error("--dataset values must be distinct")

    try:
        if not args.until_complete:
            if len(args.dataset) != 1:
                p.error("multiple --dataset values require --until-complete")
            label = run_fleet(args, args.dataset[0])
            if not args.skip_final_teardown:
                print(f"=== [{time.strftime('%H:%M:%S')}] Tearing down {label}...")
                launch_session._teardown(label)
            return
        labels = run_until_complete(args)
        if not args.skip_final_teardown:
            for label in labels:
                print(f"=== [{time.strftime('%H:%M:%S')}] Tearing down {label}...")
                launch_session._teardown(label)
    except QuotaExceeded as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(QUOTA_EXIT_CODE) from exc
    except FleetRunError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
