"""provisioning/workspaces.py — per-Terraform-workspace AWS account/namespace/
profile config, read from provisioning/workspaces.json (gitignored; copy from
workspaces.json.example).

This module is the single place that mapping lives. provision.py calls
`current_workspace_config()` before every terraform invocation instead of
inheriting whatever `.env` happens to have exported.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional, TypedDict

_DIR = Path(__file__).parent
_WORKSPACES_JSON = _DIR / "workspaces.json"
TERRAFORM_DIR = _DIR.parent / "terraform"


class WorkspaceConfig(TypedDict, total=False):
    account_id: str
    namespace: str
    aws_profile: Optional[str]
    aws_region: str
    security_group_id: str


def _load() -> dict[str, WorkspaceConfig]:
    if not _WORKSPACES_JSON.exists():
        return {}
    return json.loads(_WORKSPACES_JSON.read_text())


def current_workspace() -> str:
    """The Terraform workspace `terraform` commands in TERRAFORM_DIR will
    actually run against right now — same authority provision.py's own `_tf()`
    subprocess calls answer to, so this never gets out of sync with them."""
    result = subprocess.run(
        ["terraform", "workspace", "show"],
        cwd=TERRAFORM_DIR, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"terraform workspace show failed:\n{result.stderr}")
    return result.stdout.strip()


def current_workspace_config() -> WorkspaceConfig:
    """This workspace's entry from workspaces.json, or an empty config
    (namespace="", no account guard, no profile override) if the workspace
    isn't listed — the same "unconfigured = old, unguarded behaviour" default
    terraform/main.tf's own var.workspace_account_ids docstring describes,
    so an workspace nobody has registered yet doesn't hard-fail, it just
    isn't protected against the name-collision this module exists to avoid.

    `aws_region` and `security_group_id` have no defaults and are required.
    """
    ws = current_workspace()
    cfg = _load().get(ws, WorkspaceConfig(account_id="", namespace="", aws_profile=None))
    missing = [key for key in ("aws_region", "security_group_id") if not cfg.get(key)]
    if missing:
        raise RuntimeError(
            f"Terraform workspace {ws!r} is missing {', '.join(repr(key) for key in missing)} "
            f"in {_WORKSPACES_JSON} — add them (see workspaces.json.example)."
        )
    return cfg


def workspace_account_ids() -> dict[str, str]:
    """workspace -> AWS account id, for every workspace that declares one —
    the full map TF_VAR_workspace_account_ids needs (terraform/main.tf reads
    it as `...[terraform.workspace]`, so every workspace that might get
    selected has to be present, not just the current one)."""
    return {ws: cfg["account_id"] for ws, cfg in _load().items() if cfg.get("account_id")}


__all__ = ["WorkspaceConfig", "current_workspace", "current_workspace_config", "workspace_account_ids"]
