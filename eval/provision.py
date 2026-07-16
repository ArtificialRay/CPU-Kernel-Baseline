"""
eval/provision.py — standalone Terraform lifecycle wrapper for Arm EC2
instances (Graviton3/4). Provisions an instance, waits for it to be ready,
rsyncs source, installs deps, and (optionally) builds a dataset's native
lib. Writes/reads a single shared config file, eval/eval_config.json.

Standalone script — nothing else in this repo imports from this module.
Callers that need an instance (eval/run_benchmark.py,
skills/launch/launch_session.py, scripts/gen-workload/collect_workloads_llm.py)
invoke it as a subprocess and then read eval/eval_config.json themselves
for host/user/key_file. This is what keeps skills/launch/ (which must have
zero Python imports from eval/ — see skills/README.md) and eval/ able to
share one provisioning script and one source of truth for "what's running"
without either importing the other.

Usage:
    python eval/provision.py --isa sve2
    # Reuses a reachable instance for that ISA tier if eval_config.json has
    # one recorded, otherwise runs terraform apply for a fresh one. To force
    # a genuinely new instance: `--teardown` first, then provision again.

    python eval/provision.py --teardown
    python eval/provision.py --status
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.remote import InstanceHandle

REPO_ROOT = Path(__file__).parent.parent
TERRAFORM_DIR = REPO_ROOT / "terraform"
EVAL_CONFIG_PATH = REPO_ROOT / "eval" / "eval_config.json"
DATASET_BUILDS_PATH = REPO_ROOT / "eval" / "dataset_builds.json"

# Map ISA targets to instance types
ISA_INSTANCE_MAP = {
    "neon": "c7g.large",
    "sve": "c7g.large",
    "sve2": "c8g.large",
    "sme2": "c8g.large",
}


def _tier_for_instance_type(instance_type: str) -> str:
    return "c8g" if "c8g" in instance_type else "c7g"


def _tf(*args, capture: bool = False) -> subprocess.CompletedProcess:
    cmd = ["terraform"] + list(args)
    return subprocess.run(
        cmd,
        cwd=TERRAFORM_DIR,
        capture_output=capture,
        text=True,
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


def provision(instance_type: str = "c7g.large", initial_build: str = "", dataset: str = "") -> InstanceHandle:
    """
    Run terraform apply to provision an instance. Blocks until SSH is available,
    rsyncs source, installs deps, and runs dataset-specific build steps.

    Unconditional — always runs terraform apply (idempotent against unchanged
    Terraform state) even if a reachable instance for this tier is already up.
    Most callers want get_or_provision() instead; this is the raw primitive.

    Args:
        instance_type: EC2 instance type string (e.g. "c7g.large", "c8g.large", "c8g.xlarge")
        initial_build: make target for initial build, e.g. "c-scalar". Empty = skip.
        dataset: Dataset name (e.g. "ncnn") — triggers build steps from dataset_builds.json.
    """
    is_c8g = "c8g" in instance_type
    print(f"[provision] Provisioning {instance_type} via Terraform...")

    if is_c8g:
        # c8g has its own fixed resource block — pass instance type as a variable.
        result = _tf("apply", "-auto-approve",
                     f"-var=instance_type={instance_type}",
                     "-target=aws_instance.c8g",
                     "-target=null_resource.deploy_c8g")
    else:
        skip_build = initial_build == ""
        vars = [
            f"-var=instance_type={instance_type}",
            f"-var=skip_initial_build={'true' if skip_build else 'false'}",
        ]
        if not skip_build:
            vars.append(f"-var=build_target={initial_build}")
        # Scope to the c7g instance + its deploy resource only. Without -target this
        # runs a full-config apply that recreates the c8g instance (shared state) —
        # so provisioning a c7g while a c8g is up would destroy the c8g mid-run.
        result = _tf("apply", "-auto-approve", *vars,
                     "-target=aws_instance.kernel_testing",
                     "-target=null_resource.deploy")

    if result.returncode != 0:
        raise RuntimeError("terraform apply failed")

    outputs = _tf_output()
    if is_c8g:
        host = outputs["c8g_public_ip"]["value"]
        instance_id = outputs.get("c8g_instance_id", {}).get("value")
    else:
        host = outputs["instance_public_ip"]["value"]
        instance_id = outputs.get("instance_id", {}).get("value")
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
        excludes=["build", ".git", "terraform", "generations", "results",
                  "notebooks", "agent-runs", "agent-runs-mcp", "agent-runs-nanobot",
                  "__pycache__", "*.pyc"],
    )

    _install_deps(handle)
    if dataset:
        ensure_dataset_ready(handle, dataset)

    _save_config(handle)
    print(f"[provision] Done. SSH: ssh -i {key_file} ubuntu@{host}")
    return handle


def teardown():
    """Run terraform destroy to terminate the instance."""
    print("[teardown] Running terraform destroy...")
    result = _tf("destroy", "-auto-approve")
    if result.returncode != 0:
        raise RuntimeError("terraform destroy failed")
    if EVAL_CONFIG_PATH.exists():
        config = json.loads(EVAL_CONFIG_PATH.read_text())
        # Clear host entries but keep structure
        for tier in config.get("instances", {}):
            config["instances"][tier]["host"] = ""
        EVAL_CONFIG_PATH.write_text(json.dumps(config, indent=2))
    print("[teardown] Instance terminated.")


def get_running_instance(instance_type: str) -> InstanceHandle | None:
    """
    Return a handle to a running instance for the tier `instance_type`
    belongs to (c7g/c8g), if configured. Reads from eval_config.json.
    """
    if not EVAL_CONFIG_PATH.exists():
        return None
    config = json.loads(EVAL_CONFIG_PATH.read_text())
    tier = _tier_for_instance_type(instance_type)
    inst = config.get("instances", {}).get(tier, {})
    host = inst.get("host", "")
    if not host:
        return None
    return InstanceHandle(
        host=host,
        user=inst.get("user", "ubuntu"),
        key_file=inst.get("key_file", "~/.ssh/id_rsa"),
        instance_type=instance_type,
    )


def get_or_provision(instance_type: str, dataset: str = "") -> InstanceHandle:
    """
    Return an existing reachable instance for this tier, or provision a new one.

    Args:
        instance_type: EC2 instance type, e.g. "c7g.large", "c8g.xlarge".
        dataset: Dataset name — ensured ready on the returned instance either way.
    """
    handle = get_running_instance(instance_type)
    if handle and _is_reachable(handle):
        print(f"[provision] Reusing existing instance at {handle.host}")
        if dataset:
            ensure_dataset_ready(handle, dataset)
        return handle
    return provision(instance_type, dataset=dataset)


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


def _save_config(handle: InstanceHandle):
    config = {}
    if EVAL_CONFIG_PATH.exists():
        config = json.loads(EVAL_CONFIG_PATH.read_text())

    tier = _tier_for_instance_type(handle.instance_type)
    config.setdefault("instances", {})
    config["instances"][tier] = {
        "host": handle.host,
        "user": handle.user,
        "key_file": handle.key_file,
    }
    EVAL_CONFIG_PATH.write_text(json.dumps(config, indent=2))


def status():
    """Print current instance status from eval_config.json."""
    if not EVAL_CONFIG_PATH.exists():
        print("No eval_config.json found. Run provision first.")
        return
    config = json.loads(EVAL_CONFIG_PATH.read_text())
    for tier, inst in config.get("instances", {}).items():
        host = inst.get("host", "")
        if not host:
            print(f"  {tier}: not provisioned")
            continue
        handle = InstanceHandle(host=host, user=inst["user"],
                                key_file=inst["key_file"], instance_type=tier)
        reachable = _is_reachable(handle)
        status_str = "reachable" if reachable else "UNREACHABLE"
        print(f"  {tier}: {host} — {status_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Provision/teardown Arm EC2 instances")
    parser.add_argument("--instance", default=None,
                        help="EC2 instance type override (e.g. c8g.xlarge). "
                             "Defaults to ISA_INSTANCE_MAP value when --isa is set.")
    parser.add_argument("--isa", help="ISA target (neon/sve/sve2/sme2)")
    parser.add_argument("--teardown", action="store_true", help="Destroy the instance")
    parser.add_argument("--status", action="store_true", help="Show instance status")
    parser.add_argument("--initial-build", default="",
                        help="Run make <target> after provision, only when provisioning "
                             "a fresh instance (default: skip)")
    parser.add_argument("--dataset", default="",
                        help="Dataset to build after provisioning (e.g. ncnn). "
                             "Default: skip — instance will lack that dataset's build artifacts.")
    args = parser.parse_args()

    if args.status:
        status()
    elif args.teardown:
        teardown()
    else:
        instance_type = args.instance or (ISA_INSTANCE_MAP.get(args.isa, "c7g.large") if args.isa else "c7g.large")
        # Reuse a reachable instance for this tier if one's already up; otherwise
        # provision a fresh one. To force a genuinely new instance, run with
        # --teardown first.
        handle = get_running_instance(instance_type)
        if handle and _is_reachable(handle):
            print(f"[provision] Reusing existing instance at {handle.host}")
            if args.dataset:
                ensure_dataset_ready(handle, args.dataset)
        else:
            handle = provision(instance_type, args.initial_build, dataset=args.dataset)
        print(f"\nInstance handle: {handle}")
        print(f"host={handle.host} user={handle.user} key_file={handle.key_file} "
              f"instance_type={handle.instance_type}")
