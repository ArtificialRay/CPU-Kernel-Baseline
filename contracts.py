"""Kernel-session contracts shared by eval/, mcp_app/, skills/, and bench/.

Single source of truth for naming/mapping conventions that every dataset's
AgentTools (eval/agent_tools/*) or KernelSession (mcp_app/agent_tools/*)
implementation must agree on — which file an agent's kernel is always named,
which Solution.author a dataset's reference-scalar/baseline solution lives
under, and which compile flags + EC2 instance type an ISA maps to. Backed by
config/kernel_contracts.yaml; loaded once and cached.

Lives at the repo root (not inside bench/) because it's consumed by every
top-level package equally — bench/, eval/, mcp_app/, skills/ — none of which
should appear to "own" it. mcp_app/README.md's "zero coupling to eval/ or
skills/" boundary is preserved: this module has no dependency on any of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent
_YAML_PATH = REPO_ROOT / "config" / "kernel_contracts.yaml"


@dataclass(frozen=True)
class IsaSpec:
    march: str
    features: list[str]
    labels: list[str]
    instance_type: str


@lru_cache(maxsize=1)
def _load() -> dict:
    with _YAML_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _contracts() -> dict:
    return _load()["contracts"]


def _isa_table() -> dict[str, IsaSpec]:
    return {
        isa: IsaSpec(
            march=spec["march"],
            features=list(spec["features"]),
            labels=list(spec["labels"]),
            instance_type=spec["instance_type"],
        )
        for isa, spec in _load()["isa"].items()
    }


AGENT_KERNEL_FILENAME: str = _contracts()["agent_kernel_filename"]
REFERENCE_SCALAR_FILENAME: str = _contracts()["reference_scalar_filename"]
REFERENCE_SCALAR_AUTHORS: dict[str, str] = dict(_load()["reference_scalar_authors"])
BASELINE_AUTHORS: dict[str, str] = dict(_load()["baseline_authors"])
ISA_TABLE: dict[str, IsaSpec] = _isa_table()

# isa -> EC2 instance type, the subset of ISA_TABLE most callers actually need.
ISA_INSTANCE_MAP: dict[str, str] = {isa: spec.instance_type for isa, spec in ISA_TABLE.items()}

__all__ = [
    "IsaSpec",
    "AGENT_KERNEL_FILENAME",
    "REFERENCE_SCALAR_FILENAME",
    "REFERENCE_SCALAR_AUTHORS",
    "BASELINE_AUTHORS",
    "ISA_TABLE",
    "ISA_INSTANCE_MAP",
]
