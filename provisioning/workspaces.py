"""Load the one AWS/Terraform workspace selected for this run.

``provisioning/workspaces.json`` is intentionally a single-entry file.  The
entry name is the workspace to use, and the entry value contains its AWS
configuration.  This avoids silently selecting Terraform's ``default``
workspace because of an unrelated environment variable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, TypedDict

_DIR = Path(__file__).parent
_WORKSPACES_JSON = _DIR / "workspaces.json"


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
    """Return the sole workspace configured for this run."""
    configs = _load()
    if len(configs) != 1:
        raise RuntimeError(
            f"{_WORKSPACES_JSON} must contain exactly one current workspace entry; "
            f"found {len(configs)}. Remove stale workspace entries."
        )
    configured = next(iter(configs))
    selected = os.environ.get("TF_WORKSPACE")
    if selected and selected != configured:
        raise RuntimeError(
            f"TF_WORKSPACE={selected!r} conflicts with the configured current "
            f"workspace {configured!r} in {_WORKSPACES_JSON}."
        )
    return configured


def current_workspace_config() -> WorkspaceConfig:
    """Return the configuration for the one workspace selected for this run.

    ``aws_region`` and ``security_group_id`` have no defaults and are required.
    """
    ws = current_workspace()
    cfg = _load()[ws]
    missing = [key for key in ("aws_region", "security_group_id") if not cfg.get(key)]
    if missing:
        raise RuntimeError(
            f"Terraform workspace {ws!r} is missing {', '.join(repr(key) for key in missing)} "
            f"in {_WORKSPACES_JSON} — add them (see workspaces.json.example)."
        )
    return cfg


def workspace_account_ids() -> dict[str, str]:
    """Return the account guard for the one configured workspace."""
    return {ws: cfg["account_id"] for ws, cfg in _load().items() if cfg.get("account_id")}


__all__ = ["WorkspaceConfig", "current_workspace", "current_workspace_config", "workspace_account_ids"]
