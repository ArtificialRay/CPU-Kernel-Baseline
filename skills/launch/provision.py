"""provision — Terraform lifecycle wrapper for skills/launch's own instance
tracking. Duplicate of eval/provision.py's orchestration code (zero imports
from eval/), sharing the same physical `<repo_root>/terraform/` state (one
set of cloud resources, not two) but NOT eval/eval_config.json — this module
tracks "what's up" in its own launch_config.json, so an instance provisioned
via eval/provision.py won't show up in get_or_provision() here and vice
versa. Run `status()` if unsure what's already up.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from remote import RemoteTarget

REPO_ROOT = Path(__file__).parent.parent.parent
TERRAFORM_DIR = REPO_ROOT / "terraform"
LAUNCH_CONFIG_PATH = Path(__file__).parent / "launch_config.json"

# Own copy of eval/provision.py's ISA_INSTANCE_MAP — small (4 rows), rarely
# changes, kept in sync by hand rather than generated (same tradeoff as the
# dataset/baseline_author/isa table in SKILL.md).
ISA_INSTANCE_MAP = {
    "neon": "c7g.large",
    "sve": "c7g.large",
    "sve2": "c8g.large",
    "sme2": "c8g.large",
}

DEFAULT_RSYNC_EXCLUDES = [
    "build", ".git", "terraform", "generations", "results", "notebooks",
    "agent-runs", "agent-runs-mcp", "agent-runs-nanobot", "__pycache__", "*.pyc",
]


@dataclass(frozen=True)
class ProvisionedInstance:
    target: RemoteTarget
    instance_type: str
    instance_id: Optional[str] = None


def _tf(*args: str, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["terraform", *args], cwd=TERRAFORM_DIR, capture_output=capture, text=True,
    )


def _tf_output() -> dict:
    result = _tf("output", "-json", capture=True)
    if result.returncode != 0:
        raise RuntimeError(f"terraform output failed:\n{result.stderr}")
    return json.loads(result.stdout)


def _install_deps(target: RemoteTarget) -> None:
    """Install system/Python deps a bare Graviton instance needs before it can
    build anything under bench/ or mcp_app/ (clang-18, cmake, libomp) or run a
    long-lived mcp_app.server session without SSH getting killed mid-run by
    Ubuntu's daily unattended-upgrades timer.
    """
    steps = [
        (
            "disable unattended-upgrades",
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
        print(f"[launch/provision] Installing {label}...")
        rc, _, err = target.run(cmd, timeout=timeout)
        if rc != 0:
            print(f"[launch/provision]   WARNING: {label} failed: {err[:200]}")


def _wait_for_ssh(target: RemoteTarget, max_wait: int = 300, interval: int = 10) -> None:
    deadline = time.time() + max_wait
    while time.time() < deadline:
        if _is_reachable(target):
            return
        print(f"[launch/provision]   Waiting for SSH... (retry in {interval}s)")
        time.sleep(interval)
    raise TimeoutError(f"SSH not available on {target.host} after {max_wait}s")


def _is_reachable(target: RemoteTarget) -> bool:
    try:
        rc, _, _ = target.run("echo ok", timeout=15)
        return rc == 0
    except Exception:
        return False


def _load_config() -> dict:
    if not LAUNCH_CONFIG_PATH.exists():
        return {}
    return json.loads(LAUNCH_CONFIG_PATH.read_text())


def _save_instance(instance: ProvisionedInstance) -> None:
    config = _load_config()
    tier = "c8g" if "c8g" in instance.instance_type else "c7g"
    config.setdefault("instances", {})
    config["instances"][tier] = {
        "host": instance.target.host,
        "user": instance.target.user,
        "key_file": instance.target.key_file,
        "instance_type": instance.instance_type,
        "instance_id": instance.instance_id,
    }
    LAUNCH_CONFIG_PATH.write_text(json.dumps(config, indent=2))


def provision(
    instance_type: str = "c7g.large", *, dataset: str = "", local_repo_dir: Optional[str | Path] = None,
) -> ProvisionedInstance:
    """Terraform-apply an instance, wait for SSH, rsync this repo, install
    build deps, and (if `dataset` is given) build that dataset's native lib.

    `local_repo_dir` defaults to this repo's own root (REPO_ROOT) — override
    only if you're driving this from a different checkout than the one this
    file lives in.
    """
    is_c8g = "c8g" in instance_type
    print(f"[launch/provision] Provisioning {instance_type} via Terraform...")

    if is_c8g:
        result = _tf("apply", "-auto-approve",
                     f"-var=instance_type={instance_type}",
                     "-target=aws_instance.c8g",
                     "-target=null_resource.deploy_c8g")
    else:
        result = _tf("apply", "-auto-approve",
                     f"-var=instance_type={instance_type}",
                     "-var=skip_initial_build=true",
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

    target = RemoteTarget(host=host, user="ubuntu", key_file=key_file)

    print(f"[launch/provision] Instance ready at {host}, waiting for SSH...")
    _wait_for_ssh(target)

    print(f"[launch/provision] Rsyncing repo to {host}:~/arm-bench/...")
    target.rsync_to(str(local_repo_dir or REPO_ROOT), "~/arm-bench", excludes=DEFAULT_RSYNC_EXCLUDES)

    _install_deps(target)

    if dataset:
        from launch_session import ensure_dataset_ready
        ensure_dataset_ready(target, dataset)

    instance = ProvisionedInstance(target=target, instance_type=instance_type, instance_id=instance_id)
    _save_instance(instance)
    print(f"[launch/provision] Done. SSH: ssh -i {key_file} ubuntu@{host}")
    return instance


def teardown() -> None:
    """Terraform-destroy the instance(s). Shares state with eval/provision.py
    --teardown — this tears down the same physical instances, regardless of
    which side provisioned them.
    """
    print("[launch/provision] Running terraform destroy...")
    result = _tf("destroy", "-auto-approve")
    if result.returncode != 0:
        raise RuntimeError("terraform destroy failed")
    config = _load_config()
    for tier in config.get("instances", {}):
        config["instances"][tier]["host"] = ""
    if config:
        LAUNCH_CONFIG_PATH.write_text(json.dumps(config, indent=2))
    print("[launch/provision] Instance(s) terminated.")


def get_running_instance(isa: str) -> Optional[ProvisionedInstance]:
    """Return a handle to a running instance for `isa`, per *this module's own*
    launch_config.json — instances provisioned via eval/provision.py are not
    visible here (see module docstring)."""
    config = _load_config()
    tier = "c8g" if isa in ("sve2", "sme2") else "c7g"
    inst = config.get("instances", {}).get(tier)
    if not inst or not inst.get("host"):
        return None
    return ProvisionedInstance(
        target=RemoteTarget(host=inst["host"], user=inst.get("user", "ubuntu"),
                             key_file=inst.get("key_file", "~/.ssh/id_rsa")),
        instance_type=inst.get("instance_type", ISA_INSTANCE_MAP.get(isa, "c7g.large")),
        instance_id=inst.get("instance_id"),
    )


def get_or_provision(
    isa: str, *, instance_type: Optional[str] = None, dataset: str = "",
    local_repo_dir: Optional[str | Path] = None,
) -> ProvisionedInstance:
    """Reuse an already-up-and-reachable instance for `isa` if this module
    provisioned one earlier; otherwise provision a fresh one."""
    existing = get_running_instance(isa)
    if existing and _is_reachable(existing.target):
        print(f"[launch/provision] Reusing existing instance at {existing.target.host}")
        return existing
    instance_type = instance_type or ISA_INSTANCE_MAP.get(isa, "c7g.large")
    return provision(instance_type, dataset=dataset, local_repo_dir=local_repo_dir)


def status() -> None:
    config = _load_config()
    if not config.get("instances"):
        print("No instances in launch_config.json. Run `provision` first.")
        return
    for tier, inst in config["instances"].items():
        host = inst.get("host", "")
        if not host:
            print(f"  {tier}: not provisioned")
            continue
        target = RemoteTarget(host=host, user=inst.get("user", "ubuntu"),
                               key_file=inst.get("key_file", "~/.ssh/id_rsa"))
        reachable = "reachable" if _is_reachable(target) else "UNREACHABLE"
        print(f"  {tier}: {host} — {reachable}")


__all__ = [
    "ProvisionedInstance", "ISA_INSTANCE_MAP", "provision", "teardown",
    "get_running_instance", "get_or_provision", "status",
]
