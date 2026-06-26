"""
eval/provision.py — Terraform lifecycle wrapper for arm-bench benchmark.

Provisions an Arm EC2 instance (Graviton3/4), waits for it to be ready,
rsyncs source, and optionally does an initial build. Returns an InstanceHandle
that the eval tools use for SSH access.

Usage:
    python eval/provision.py --instance c7g.large
    python eval/provision.py --teardown
    python eval/provision.py --status
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

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


@dataclass
class InstanceHandle:
    host: str
    user: str
    key_file: str
    instance_type: str
    instance_id: str | None = None

    def ssh_base_args(self) -> list[str]:
        key = os.path.expanduser(self.key_file)
        return [
            "-i", key,
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=10",
        ]

    def ssh_cmd(self, remote_cmd: str) -> list[str]:
        return ["ssh"] + self.ssh_base_args() + [f"{self.user}@{self.host}", remote_cmd]

    def run(self, remote_cmd: str, timeout: int = 120) -> tuple[int, str, str]:
        result = subprocess.run(
            self.ssh_cmd(remote_cmd),
            capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr

    def upload_file(self, local_path: str, remote_path: str):
        key = os.path.expanduser(self.key_file)
        subprocess.run([
            "scp",
            "-i", key,
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            str(local_path),
            f"{self.user}@{self.host}:{remote_path}",
        ], check=True, capture_output=True)

    def rsync_to(self, local_dir: str, remote_dir: str, excludes: list[str] | None = None):
        key = os.path.expanduser(self.key_file)
        cmd = [
            "rsync", "-avz",
            "-e", f"ssh -i {key} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null",
        ]
        for exc in (excludes or []):
            cmd += ["--exclude", exc]
        cmd += [str(local_dir) + "/", f"{self.user}@{self.host}:{remote_dir}/"]
        subprocess.run(cmd, check=True, capture_output=True)


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



def _run_dataset_build(handle: InstanceHandle, dataset: str) -> None:
    """Run dataset-specific build steps on the remote, as defined in dataset_builds.json."""
    if not DATASET_BUILDS_PATH.exists():
        return
    steps = json.load(DATASET_BUILDS_PATH.open()).get(dataset, [])
    if not steps:
        return
    print(f"[provision] Building dataset '{dataset}' ({len(steps)} step(s))...")
    for step in steps:
        label = step["label"]
        print(f"[provision]   {label}...")
        rc, _, err = handle.run(step["cmd"], timeout=step.get("timeout", 300))
        if rc != 0:
            print(f"[provision]   WARNING: {label} failed: {err[:200]}")


def _install_deps(handle: InstanceHandle) -> None:
    """Install system and Python dependencies on the remote instance."""
    steps = [
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

    Args:
        instance_type: EC2 instance type string (e.g. "c7g.large", "c8g.large")
        initial_build: make target for initial build, e.g. "c-scalar". Empty = skip.
        dataset: Dataset name (e.g. "ncnn") — triggers build steps from dataset_builds.json.
    """
    is_c8g = "c8g" in instance_type
    print(f"[provision] Provisioning {instance_type} via Terraform...")

    if is_c8g:
        # c8g has its own fixed resource block — target it directly
        result = _tf("apply", "-auto-approve",
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
        result = _tf("apply", "-auto-approve", *vars)

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
                  "__pycache__", "*.pyc", ".venv"], # avoid sync local venv to remote instance
    )

    _install_deps(handle)
    if dataset:
        _run_dataset_build(handle, dataset)

    _save_config(handle)
    print(f"[provision] Done. SSH: ssh -i {key_file} ubuntu@{host}")
    return handle


def provision_n(
    n: int,
    instance_type: str = "c8g.large",
    initial_build: str = "",
    dataset: str = "",
) -> list[InstanceHandle]:
    """Provision n instances of instance_type.

    The first instance is created via Terraform (existing path). Additional
    instances are launched via boto3 by cloning the primary's AMI, key, SG,
    and subnet — so no separate Terraform config is needed.

    All extra instances are set up in parallel (rsync + deps + dataset build).
    All n handles are written to eval_config.json as a list for future re-use.

    Requires:
      - AWS credentials available locally (boto3 reads ~/.aws/credentials or env)
      - Primary Terraform instance must expose instance_id in terraform outputs
    """
    import concurrent.futures

    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    # Step 1: provision primary via Terraform
    primary = provision(instance_type, initial_build=initial_build, dataset=dataset)

    if n == 1:
        return [primary]

    if not primary.instance_id:
        raise RuntimeError(
            "Cannot clone instances: primary has no instance_id. "
            "Check that terraform output includes instance_id / c8g_instance_id."
        )

    # Step 2: describe primary to get launch config for cloning
    import boto3
    ec2 = boto3.client("ec2")

    info = ec2.describe_instances(InstanceIds=[primary.instance_id])
    src = info["Reservations"][0]["Instances"][0]
    ami_id   = src["ImageId"]
    key_name = src["KeyName"]
    sg_ids   = [sg["GroupId"] for sg in src["SecurityGroups"]]
    subnet_id = src["SubnetId"]

    n_extra = n - 1
    print(f"[provision_n] Launching {n_extra} extra {instance_type} via boto3...")
    resp = ec2.run_instances(
        ImageId=ami_id,
        InstanceType=instance_type,
        KeyName=key_name,
        SecurityGroupIds=sg_ids,
        SubnetId=subnet_id,
        MinCount=n_extra,
        MaxCount=n_extra,
        TagSpecifications=[{
            "ResourceType": "instance",
            "Tags": [{"Key": "Name", "Value": "arm-bench-worker"}],
        }],
    )
    new_ids = [i["InstanceId"] for i in resp["Instances"]]

    print(f"[provision_n] Waiting for {n_extra} instance(s) to reach running state...")
    ec2.get_waiter("instance_running").wait(InstanceIds=new_ids)

    extra_handles: list[InstanceHandle] = []
    for inst_id in new_ids:
        desc = ec2.describe_instances(InstanceIds=[inst_id])
        ip = desc["Reservations"][0]["Instances"][0]["PublicIpAddress"]
        extra_handles.append(InstanceHandle(
            host=ip,
            user=primary.user,
            key_file=primary.key_file,
            instance_type=instance_type,
            instance_id=inst_id,
        ))

    # Step 3: parallel SSH wait + rsync + deps on extra instances
    def _setup(h: InstanceHandle) -> None:
        print(f"[provision_n] Setting up {h.host} ...")
        _wait_for_ssh(h)
        h.rsync_to(
            str(REPO_ROOT), "~/arm-bench",
            excludes=["build", ".git", "terraform", "generations", "results",
                      "__pycache__", "*.pyc", ".venv"],
        )  # avoid sync local venv to remote instance
        _install_deps(h)
        if dataset:
            _run_dataset_build(h, dataset)
        print(f"[provision_n] {h.host} ready.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(extra_handles)) as pool:
        for fut in concurrent.futures.as_completed(
            [pool.submit(_setup, h) for h in extra_handles]
        ):
            fut.result()  # re-raise if setup failed

    all_handles = [primary] + extra_handles
    _save_config_multi(all_handles)
    print(f"[provision_n] {n} instance(s) ready: {[h.host for h in all_handles]}")
    return all_handles


def _save_config_multi(handles: list[InstanceHandle]) -> None:
    """Write all handles for the same tier as a list in eval_config.json."""
    if not handles:
        return
    config = {}
    if EVAL_CONFIG_PATH.exists():
        config = json.loads(EVAL_CONFIG_PATH.read_text())

    tier = "c8g" if "c8g" in handles[0].instance_type else "c7g"
    config.setdefault("instances", {})

    entries = []
    for h in handles:
        entry: dict = {"host": h.host, "user": h.user, "key_file": h.key_file}
        if h.instance_id:
            entry["instance_id"] = h.instance_id
        entries.append(entry)

    # Keep single-dict format when n=1 for backward compat; list when n>1.
    config["instances"][tier] = entries if len(entries) > 1 else entries[0]
    EVAL_CONFIG_PATH.write_text(json.dumps(config, indent=2))


def teardown():
    """Run terraform destroy to terminate the Terraform-managed instance."""
    print("[teardown] Running terraform destroy...")
    result = _tf("destroy", "-auto-approve")
    if result.returncode != 0:
        raise RuntimeError("terraform destroy failed")
    if EVAL_CONFIG_PATH.exists():
        config = json.loads(EVAL_CONFIG_PATH.read_text())
        # Clear host entries but keep structure; handle both dict and list formats.
        for tier in config.get("instances", {}):
            val = config["instances"][tier]
            if isinstance(val, list):
                for entry in val:
                    entry["host"] = ""
            elif isinstance(val, dict):
                val["host"] = ""
        EVAL_CONFIG_PATH.write_text(json.dumps(config, indent=2))
    print("[teardown] Instance terminated.")


def get_running_instances(isa: str, n: int = 1) -> list[InstanceHandle]:
    """Return up to n InstanceHandles for running instances of the given ISA.

    Reads from eval_config.json. The value for each tier can be a single dict
    (legacy, single-instance) or a list of dicts (multi-instance). Entries with
    an empty host are skipped.

    Multi-instance example (eval_config.json):
        {"instances": {"c8g": [
            {"host": "1.2.3.4", "user": "ubuntu", "key_file": "~/.ssh/id_rsa"},
            {"host": "5.6.7.8", "user": "ubuntu", "key_file": "~/.ssh/id_rsa"}
        ]}}
    """
    if not EVAL_CONFIG_PATH.exists():
        return []
    config = json.loads(EVAL_CONFIG_PATH.read_text())
    tier = "c8g" if isa in ("sve2", "sme2") else "c7g"
    raw = config.get("instances", {}).get(tier)
    if raw is None:
        return []

    instance_type = ISA_INSTANCE_MAP.get(isa, "c7g.large")
    entries = raw if isinstance(raw, list) else [raw]

    handles = []
    for entry in entries[:n]:
        host = entry.get("host", "")
        if not host:
            continue
        handles.append(InstanceHandle(
            host=host,
            user=entry.get("user", "ubuntu"),
            key_file=entry.get("key_file", "~/.ssh/id_rsa"),
            instance_type=instance_type,
            instance_id=entry.get("instance_id"),
        ))
    return handles


def get_running_instance(isa: str) -> InstanceHandle | None:
    """Return a handle to a single running instance for the given ISA, if configured."""
    handles = get_running_instances(isa, 1)
    return handles[0] if handles else None


def get_or_provision(isa: str) -> InstanceHandle:
    """
    Return an existing running instance or provision a new one.
    """
    handle = get_running_instance(isa)
    if handle and _is_reachable(handle):
        print(f"[provision] Reusing existing instance at {handle.host}")
        return handle
    instance_type = ISA_INSTANCE_MAP.get(isa, "c7g.large")
    return provision(instance_type)


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

    tier = "c8g" if "c8g" in handle.instance_type else "c7g"
    config.setdefault("instances", {})
    entry: dict = {"host": handle.host, "user": handle.user, "key_file": handle.key_file}
    if handle.instance_id:
        entry["instance_id"] = handle.instance_id
    # Single-instance dict format; users may manually convert to list for multi-instance.
    config["instances"][tier] = entry
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
    parser.add_argument("--instance", default="c7g.large",
                        help="EC2 instance type (default: c7g.large)")
    parser.add_argument("--isa", help="ISA target (neon/sve/sve2/sme2); overrides --instance")
    parser.add_argument("--teardown", action="store_true", help="Destroy the instance")
    parser.add_argument("--status", action="store_true", help="Show instance status")
    parser.add_argument("--initial-build", default="",
                        help="Run make <target> after provision (default: skip)")
    args = parser.parse_args()

    if args.status:
        status()
    elif args.teardown:
        teardown()
    else:
        instance_type = ISA_INSTANCE_MAP.get(args.isa, args.instance) if args.isa else args.instance
        handle = provision(instance_type, args.initial_build)
        print(f"\nInstance handle: {handle}")
