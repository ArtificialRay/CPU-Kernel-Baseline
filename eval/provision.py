"""
eval/provision.py — standalone Terraform lifecycle wrapper for Arm EC2
instances (Graviton3/4). Provisions an instance, waits for it to be ready,
rsyncs source, installs deps, and (optionally) builds a dataset's native
lib. Writes/reads a single shared config file, eval/eval_config.json.

Every instance is identified by a `label` — an arbitrary caller-chosen name,
one per concurrently-desired instance (see `default_label()` below for the
--isa/--dataset-derived default). This replaced an earlier tier-keyed
("c7g"/"c8g") design that could only ever track one instance per ISA tier —
two jobs on the same ISA (e.g. ncnn+sve and llama.cpp+sve) had no way to get
two separate instances. `label` maps directly onto terraform/main.tf's
`var.instances` map key (`aws_instance.labeled[label]`).

Standalone script — nothing else in this repo imports from this module.
Callers that need an instance (eval/run_benchmark.py,
skills/launch/launch_session.py, scripts/gen-workload/collect_workloads_llm.py)
invoke it as a subprocess and then read eval/eval_config.json themselves
for host/user/key_file. This is what keeps skills/launch/ (which must have
zero Python imports from eval/ — see skills/README.md) and eval/ able to
share one provisioning script and one source of truth for "what's running"
without either importing the other.

Usage:
    python eval/provision.py --isa sve2 --dataset ncnn
    # label defaults to "ncnn-sve2". Reuses a reachable instance under that
    # label if eval_config.json has one recorded, otherwise runs terraform
    # apply for a fresh one. To force a genuinely new instance: `--teardown
    # --label ncnn-sve2` first, then provision again.

    python eval/provision.py --teardown --label ncnn-sve2   # tear down just that one
    python eval/provision.py --teardown                     # tear down every known label
    python eval/provision.py --status
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from contracts import ISA_INSTANCE_MAP
from eval.remote import InstanceHandle

REPO_ROOT = Path(__file__).parent.parent
TERRAFORM_DIR = REPO_ROOT / "terraform"
EVAL_CONFIG_PATH = REPO_ROOT / "eval" / "eval_config.json"

# Repo-root-relative paths mcp_app/bench actually need on the remote side.
# Allow-list, not a deny-list — see InstanceHandle.rsync_to's docstring.
# TODO: fold into an env var
RSYNC_ALLOWLIST = ["bench", "bench-trace", "mcp_app", "requirements.txt","config","contracts.py"]
DATASET_BUILDS_PATH = REPO_ROOT / "eval" / "dataset_builds.json"

# Must stay shell/HCL/JSON-key/AWS-tag safe — flows into a `terraform -target`
# CLI arg, an eval_config.json dict key, and an AWS resource tag. Rejected
# outright rather than silently sanitized (see default_label()'s docstring).
LABEL_RE = re.compile(r"^[a-z0-9]([a-z0-9.-]*[a-z0-9])?$")


def _validate_label(label: str) -> str:
    if not LABEL_RE.match(label):
        raise ValueError(
            f"Invalid label {label!r} — must match {LABEL_RE.pattern} "
            "(lowercase letters/digits, '.'/'-' in the middle only)."
        )
    return label


def default_label(isa: str | None, dataset: str, instance_type: str) -> str:
    """label defaults to f'{dataset}-{isa}' when both are known — concatenating
    both, not just dataset alone, matters: two jobs on the *same* dataset but
    *different* isa (e.g. ncnn+sve vs ncnn+sve2) need distinct instances too,
    since different isa already implies a different instance_type. Falls back
    to isa alone, then to the tier implied by instance_type, when dataset/isa
    aren't given — reproducing the old one-instance-per-tier default when
    nothing more specific is available.
    """
    if dataset and isa:
        return f"{dataset}-{isa}"
    if isa:
        return isa
    return _tier_for_instance_type(instance_type)


def _tier_for_instance_type(instance_type: str) -> str:
    return "c8g" if "c8g" in instance_type else "c7g"


def _tf(*args, capture: bool = False, extra_env: dict | None = None) -> subprocess.CompletedProcess:
    cmd = ["terraform"] + list(args)
    env = {**os.environ, **extra_env} if extra_env else None
    return subprocess.run(
        cmd,
        cwd=TERRAFORM_DIR,
        capture_output=capture,
        text=True,
        env=env,
    )


def _tf_output() -> dict:
    result = _tf("output", "-json", capture=True)
    if result.returncode != 0:
        raise RuntimeError(f"terraform output failed:\n{result.stderr}")
    return json.loads(result.stdout)


def _dataset_config(dataset: str) -> dict | None:
    """Load the dataset_builds.json entry for `dataset`, or None if it has no build steps."""
    if not DATASET_BUILDS_PATH.exists():
        return None
    return json.load(DATASET_BUILDS_PATH.open()).get(dataset)


def _dataset_ready(handle: InstanceHandle, config: dict) -> bool:
    """True if `config`'s ready_check command exits 0 on the remote."""
    ready_check = config.get("ready_check")
    if not ready_check:
        return False
    rc, _, _ = handle.run(ready_check, timeout=15)
    return rc == 0


def _run_dataset_build(handle: InstanceHandle, dataset: str, config: dict) -> bool:
    """Run dataset build steps on the remote. Returns True iff every step succeeded."""
    steps = config.get("steps", [])
    if not steps:
        return True
    print(f"[provision] Building dataset {dataset!r} ({len(steps)} step(s))...")
    ok = True
    for step in steps:
        label = step["label"]
        print(f"[provision]   {label}...")
        rc, _, err = handle.run(step["cmd"], timeout=step.get("timeout", 300))
        if rc != 0:
            print(f"[provision]   WARNING: {label} failed: {err[:200]}")
            ok = False
    return ok


def ensure_dataset_ready(handle: InstanceHandle, dataset: str) -> None:
    """Make sure `dataset`'s build artifacts are present on `handle`, building if needed.

    Safe to call on an instance that was provisioned without this dataset (or without
    any dataset at all) — it self-heals by building on demand. Raises RuntimeError if
    the build steps fail (or the ready_check still fails afterward), so callers don't
    silently proceed to run an agent against an instance missing what it needs.
    """
    if not dataset:
        return
    config = _dataset_config(dataset)
    if not config:
        return  # no build steps registered for this dataset (e.g. simd-loop)
    if _dataset_ready(handle, config):
        print(f"[provision] Dataset {dataset!r} already built on {handle.host}.")
        return
    print(f"[provision] Dataset {dataset!r} not ready on {handle.host}; building...")
    built = _run_dataset_build(handle, dataset, config)
    if not _dataset_ready(handle, config):
        raise RuntimeError(
            f"Dataset {dataset!r} failed to build on {handle.host}. "
            f"SSH in and check manually before running an eval."
        )
    if not built:
        print(f"[provision] Dataset {dataset!r} ready on {handle.host} "
              f"(one or more build steps reported a non-zero exit above, but "
              f"the ready_check now passes — likely a harmless re-run).")
    else:
        print(f"[provision] Dataset {dataset!r} ready on {handle.host}.")


def _install_deps(handle: InstanceHandle) -> None:
    """Install system and Python dependencies on the remote instance."""
    steps = [
        (
            "disable unattended-upgrades",
            # Ubuntu's apt-daily-upgrade.timer fires once a day at a randomized
            # time and, when it happens to include openssh-server, restarts
            # ssh.service — which kills every established SSH session (and
            # whatever long-running eval/mcp_app.server process is using it)
            # with no clean FIN the client can detect, so it hangs forever
            # instead of erroring out. agent benchmarking sessions can run for
            # many minutes, squarely in the blast radius of a daily timer, so
            # disable it up front rather than discover it mid-run.
            "sudo systemctl disable --now unattended-upgrades.service "
            "apt-daily.timer apt-daily-upgrade.timer "
            "apt-daily.service apt-daily-upgrade.service 2>/dev/null; true",
            30,
        ),
        (
            "apt packages",
            "sudo apt-get update -qq && "
            "sudo apt-get install -y -qq python3-pip clang-18 cmake libomp-18-dev",
            300,
        ),
        (
            "pip packages",
            "pip3 install --user --break-system-packages -r ~/arm-bench/requirements.txt",
            120,
        ),
        (
            "perf counters",
            "sudo sysctl -w kernel.perf_event_paranoid=1",
            10,
        ),
    ]
    for label, cmd, timeout in steps:
        print(f"[provision] Installing {label}...")
        rc, _, err = handle.run(cmd, timeout=timeout)
        if rc != 0:
            print(f"[provision] WARNING: {label} failed: {err[:200]}")


def provision(
    label: str,
    instance_type: str = "c7g.large",
    initial_build: str = "",
    dataset: str = "",
    on_demand: bool = False,
) -> InstanceHandle:
    """
    Run terraform apply to provision an instance under `label`. Blocks until
    SSH is available, rsyncs source, installs deps, and runs dataset-specific
    build steps.

    Unconditional — always runs terraform apply (idempotent against unchanged
    Terraform state) even if a reachable instance for this label is already up.
    Most callers want get_or_provision() instead; this is the raw primitive.

    Args:
        label: Arbitrary name identifying this instance — see module docstring
            and default_label(). Must match LABEL_RE.
        instance_type: EC2 instance type string (e.g. "c7g.large", "c8g.large", "c8g.xlarge")
        initial_build: make target for initial build, e.g. "c-scalar". Empty = skip.
        dataset: Dataset name (e.g. "ncnn") — triggers build steps from dataset_builds.json.
        on_demand: If True, provision on-demand instead of the default spot — AWS
            won't reclaim the instance mid-run, at a higher hourly price.
    """
    _validate_label(label)
    print(f"[provision] Provisioning {instance_type} (label={label!r}) via Terraform"
          f"{' (on-demand)' if on_demand else ' (spot)'}...")

    vars = [f"-var=on_demand={'true' if on_demand else 'false'}"]
    if initial_build:
        vars.append(f"-var=build_target={initial_build}")
    # -target scopes this apply to just this label's instance + deploy resource.
    # Without -target, an apply only sees whatever single label is in
    # TF_VAR_instances below and would treat every OTHER already-provisioned
    # label as "should be destroyed" (see terraform/main.tf's var.instances
    # docstring) — -target is what keeps concurrent labels from stepping on
    # each other.
    result = _tf(
        "apply", "-auto-approve", *vars,
        f'-target=aws_instance.labeled["{label}"]',
        f'-target=null_resource.deploy["{label}"]',
        extra_env={"TF_VAR_instances": json.dumps({label: instance_type})},
    )

    if result.returncode != 0:
        raise RuntimeError("terraform apply failed")

    outputs = _tf_output()
    host = outputs["instance_public_ips"]["value"][label]
    instance_id = outputs.get("instance_ids", {}).get("value", {}).get(label)
    key_file = outputs.get("ssh_key_path", {}).get("value", "~/.ssh/id_rsa")

    handle = InstanceHandle(
        host=host,
        user="ubuntu",
        key_file=key_file,
        instance_type=instance_type,
        instance_id=instance_id,
    )

    print(f"[provision] Instance ready at {host}, waiting for SSH...")
    _wait_for_ssh(handle)

    print(f"[provision] Rsyncing source to {host}:~/arm-bench/...")
    handle.rsync_to(
        str(REPO_ROOT),
        "~/arm-bench",
        paths=RSYNC_ALLOWLIST,
    )

    if initial_build:
        print(f"[provision] Building initial target {initial_build!r}...")
        rc, _, err = handle.run(f"cd ~/arm-bench && make {initial_build}", timeout=300)
        if rc != 0:
            print(f"[provision]   WARNING: initial build {initial_build!r} failed: {err[:200]}")

    _install_deps(handle)
    if dataset:
        ensure_dataset_ready(handle, dataset)

    _save_config(handle, label)
    print(f"[provision] Done. SSH: ssh -i {key_file} ubuntu@{host}")
    return handle


def teardown(label: str | None = None):
    """Run terraform destroy to terminate the instance(s).

    `label` given: destroys just that one instance (-target-scoped), leaving
    every other label's instance untouched. `label` omitted: tears down every
    label currently recorded in eval_config.json, one at a time — reproduces
    the old "destroy everything" behavior as an explicit opt-in rather than
    the default, now that a single instance is no longer the only thing that
    could possibly be up.
    """
    if label is None:
        if not EVAL_CONFIG_PATH.exists():
            print("[teardown] No eval_config.json found — nothing to tear down.")
            return
        config = json.loads(EVAL_CONFIG_PATH.read_text())
        labels = list(config.get("instances", {}))
        if not labels:
            print("[teardown] No labels recorded — nothing to tear down.")
            return
        for l in labels:
            teardown(l)
        return

    _validate_label(label)
    instance_type = _recorded_instance_type(label) or "c7g.large"
    print(f"[teardown] Destroying label={label!r}...")
    result = _tf(
        "destroy", "-auto-approve",
        f'-target=aws_instance.labeled["{label}"]',
        f'-target=null_resource.deploy["{label}"]',
        extra_env={"TF_VAR_instances": json.dumps({label: instance_type})},
    )
    if result.returncode != 0:
        raise RuntimeError(f"terraform destroy failed for label={label!r}")
    if EVAL_CONFIG_PATH.exists():
        config = json.loads(EVAL_CONFIG_PATH.read_text())
        if label in config.get("instances", {}):
            config["instances"][label]["host"] = ""
        EVAL_CONFIG_PATH.write_text(json.dumps(config, indent=2))
    print(f"[teardown] label={label!r} terminated.")


def _recorded_instance_type(label: str) -> str | None:
    """instance_type eval_config.json has on file for `label`, if any — needed
    so a destroy's TF_VAR_instances still resolves this label's for_each key
    even when the caller doesn't otherwise know its instance_type."""
    if not EVAL_CONFIG_PATH.exists():
        return None
    config = json.loads(EVAL_CONFIG_PATH.read_text())
    return config.get("instances", {}).get(label, {}).get("instance_type")


def get_running_instance(label: str) -> InstanceHandle | None:
    """
    Return a handle to the running instance recorded under `label`, if any.
    Reads from eval_config.json.
    """
    _validate_label(label)
    if not EVAL_CONFIG_PATH.exists():
        return None
    config = json.loads(EVAL_CONFIG_PATH.read_text())
    inst = config.get("instances", {}).get(label, {})
    host = inst.get("host", "")
    if not host:
        return None
    return InstanceHandle(
        host=host,
        user=inst.get("user", "ubuntu"),
        key_file=inst.get("key_file", "~/.ssh/id_rsa"),
        instance_type=inst.get("instance_type", "c7g.large"),
    )


def get_or_provision(
    label: str, instance_type: str, dataset: str = "", on_demand: bool = False
) -> InstanceHandle:
    """
    Return an existing reachable instance under this label, or provision a new one.

    Args:
        label: Arbitrary name identifying this instance — see module docstring.
        instance_type: EC2 instance type, e.g. "c7g.large", "c8g.xlarge".
        dataset: Dataset name — ensured ready on the returned instance either way.
        on_demand: Only takes effect when you want a long-run job.
    """
    handle = get_running_instance(label)
    if handle and _is_reachable(handle):
        print(f"[provision] Reusing existing instance at {handle.host} (label={label!r})")
        if dataset:
            ensure_dataset_ready(handle, dataset)
        return handle
    return provision(label, instance_type, dataset=dataset, on_demand=on_demand)


def _wait_for_ssh(handle: InstanceHandle, max_wait: int = 300, interval: int = 10):
    deadline = time.time() + max_wait
    while time.time() < deadline:
        if _is_reachable(handle):
            return
        print(f"  Waiting for SSH... (retry in {interval}s)")
        time.sleep(interval)
    raise TimeoutError(f"SSH not available on {handle.host} after {max_wait}s")


def _is_reachable(handle: InstanceHandle) -> bool:
    try:
        rc, _, _ = handle.run("echo ok", timeout=15)
        return rc == 0
    except Exception:
        return False


def _save_config(handle: InstanceHandle, label: str):
    config = {}
    if EVAL_CONFIG_PATH.exists():
        config = json.loads(EVAL_CONFIG_PATH.read_text())

    config.setdefault("instances", {})
    config["instances"][label] = {
        "host": handle.host,
        "user": handle.user,
        "key_file": handle.key_file,
        "instance_type": handle.instance_type,
    }
    EVAL_CONFIG_PATH.write_text(json.dumps(config, indent=2))


def status():
    """Print current instance status from eval_config.json."""
    if not EVAL_CONFIG_PATH.exists():
        print("No eval_config.json found. Run provision first.")
        return
    config = json.loads(EVAL_CONFIG_PATH.read_text())
    for label, inst in config.get("instances", {}).items():
        host = inst.get("host", "")
        instance_type = inst.get("instance_type", "?")
        if not host:
            print(f"  {label} ({instance_type}): not provisioned")
            continue
        handle = InstanceHandle(host=host, user=inst["user"],
                                key_file=inst["key_file"], instance_type=instance_type)
        reachable = _is_reachable(handle)
        status_str = "reachable" if reachable else "UNREACHABLE"
        print(f"  {label} ({instance_type}): {host} — {status_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Provision/teardown Arm EC2 instances")
    parser.add_argument("--instance", default=None,
                        help="EC2 instance type override (e.g. c8g.xlarge). "
                             "Defaults to ISA_INSTANCE_MAP value when --isa is set.")
    parser.add_argument("--isa", help="ISA target (neon/sve/sve2/sme2)")
    parser.add_argument("--label", default=None,
                        help="Name identifying this instance — one per concurrently-desired "
                             "instance. Default: f'{dataset}-{isa}' if --dataset given, else "
                             "--isa, else the tier implied by --instance. Must match "
                             f"{LABEL_RE.pattern}.")
    parser.add_argument("--teardown", action="store_true", help="Destroy the instance")
    parser.add_argument("--status", action="store_true", help="Show instance status")
    parser.add_argument("--initial-build", default="",
                        help="Run make <target> after provision, only when provisioning "
                             "a fresh instance (default: skip)")
    parser.add_argument("--dataset", default="",
                        help="Dataset to build after provisioning (e.g. ncnn). "
                             "Default: skip — instance will lack that dataset's build artifacts.")
    parser.add_argument("--on-demand", action="store_true",
                        help="Provision on-demand instead of the default spot — AWS won't "
                             "reclaim the instance mid-run, at a higher hourly price. Use "
                             "when you want to start a long-run job.")
    args = parser.parse_args()

    if args.status:
        status()
    elif args.teardown:
        # --label given: destroy just that one instance. Omitted: tear down
        # every label eval_config.json knows about (old "destroy everything"
        # behavior, kept as an explicit opt-in).
        teardown(args.label)
    else:
        instance_type = args.instance or (ISA_INSTANCE_MAP.get(args.isa, "c7g.large") if args.isa else "c7g.large")
        label = args.label or default_label(args.isa, args.dataset, instance_type)
        # Reuse a reachable instance for this label if one's already up; otherwise
        # provision a fresh one. To force a genuinely new instance, run with
        # --teardown --label <label> first.
        handle = get_running_instance(label)
        if handle and _is_reachable(handle):
            print(f"[provision] Reusing existing instance at {handle.host} (label={label!r})")
            if args.dataset:
                ensure_dataset_ready(handle, args.dataset)
        else:
            handle = provision(label, instance_type, args.initial_build, dataset=args.dataset,
                               on_demand=args.on_demand)
        print(f"\nInstance handle: {handle}")
        print(f"label={label} host={handle.host} user={handle.user} key_file={handle.key_file} "
              f"instance_type={handle.instance_type}")
