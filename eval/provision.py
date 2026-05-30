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


def provision(instance_type: str = "c7g.large", initial_build: str = "") -> InstanceHandle:
    """
    Run terraform apply to provision an instance. Blocks until SSH is available
    and source is rsynced.

    Args:
        instance_type: EC2 instance type string (e.g. "c7g.large", "c8g.large")
        initial_build: make target for initial build, e.g. "c-scalar". Empty = skip.
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
                  "__pycache__", "*.pyc"],
    )

    _save_config(handle)
    print(f"[provision] Done. SSH: ssh -i {key_file} ubuntu@{host}")
    return handle


def provision_codebase(instance_type: str = "c7g.large", initial_build: str = "", codebase: str = "") -> InstanceHandle:
    """
    Sync the codebase to the remote instance for cpu kernel codebase evaluation.

    The codebase source is expected at CPU-Kernel-Baseline/{codebase}/ relative to arm-bench's
    parent directory. It is rsynced to ~/{codebase}/ on the remote instance, which is
    the path that NCNNTools expects.

    Args:
        handle: An InstanceHandle already connected to the remote instance.
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
        result = _tf("apply", "-auto-approve",
             "-target=aws_instance.kernel_testing",
             "-target=null_resource.deploy",
             *vars)


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

    # Look for ncnn/ in the CPU-Kernel-Baseline repo next to arm-bench
    codebase_dir = REPO_ROOT.parent / "CPU-Kernel-Baseline" / codebase
    if not codebase_dir.exists():
        # Fallback: look for ncnn/ directly next to arm-bench
        codebase_dir = REPO_ROOT.parent / codebase
        if not codebase_dir.exists():
            raise FileNotFoundError(
                f"ncnn codebase not found. Looked at:\n"
                f"  {REPO_ROOT.parent / 'CPU-Kernel-Baseline' / codebase}\n"
                f"  {REPO_ROOT.parent / codebase}\n"
                f"Make sure CPU-Kernel-Baseline/{codebase} exists relative to arm-bench."
            )

    print(f"[provision_codebase] Rsyncing {codebase_dir} → {handle.host}:~/{codebase}/ ...")
    handle.rsync_to(
        str(codebase_dir),
        f"~/{codebase}",
        excludes=["build", ".git", "__pycache__", "*.o", "*.d", "*.pyc"],
    )
    _save_config(handle)
    print(f"[provision_codebase] {codebase} codebase synced.")
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


def get_running_instance(isa: str) -> InstanceHandle | None:
    """
    Return a handle to a running instance for the given ISA, if configured.
    Reads from eval_config.json.
    """
    if not EVAL_CONFIG_PATH.exists():
        return None
    config = json.loads(EVAL_CONFIG_PATH.read_text())
    tier = "c8g" if isa in ("sve2", "sme2") else "c7g"
    inst = config.get("instances", {}).get(tier, {})
    host = inst.get("host", "")
    if not host:
        return None
    return InstanceHandle(
        host=host,
        user=inst.get("user", "ubuntu"),
        key_file=inst.get("key_file", "~/.ssh/id_rsa"),
        instance_type=ISA_INSTANCE_MAP.get(isa, "c7g.large"),
    )


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
    parser.add_argument("--instance", default="c7g.large",
                        help="EC2 instance type (default: c7g.large)")
    parser.add_argument("--isa", help="ISA target (neon/sve/sve2/sme2); overrides --instance")
    parser.add_argument("--teardown", action="store_true", help="Destroy the instance")
    parser.add_argument("--status", action="store_true", help="Show instance status")
    parser.add_argument("--initial-build", default="",
                        help="Run make <target> after provision (default: skip)")
    parser.add_argument("--codebase",default="ncnn",help="CPU kernel baseline codebase selection")
    args = parser.parse_args()

    if args.status:
        status()
    elif args.teardown:
        teardown()
    elif args.codebase:
        instance_type = ISA_INSTANCE_MAP.get(args.isa, args.instance) if args.isa else args.instance
        handle = provision_codebase(instance_type, args.initial_build, args.codebase)
        print(f"\nInstance handle: {handle}")
    else:
        instance_type = ISA_INSTANCE_MAP.get(args.isa, args.instance) if args.isa else args.instance
        handle = provision(instance_type, args.initial_build)
        print(f"\nInstance handle: {handle}")
