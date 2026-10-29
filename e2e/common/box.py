"""Finding the Graviton box a step is to drive over SSH.

Nothing in e2e starts a box: it is provisioned with provisioning/provision.py (or by
bench_fleet.py) and registered in provisioning/eval_config.json under a label."""
from __future__ import annotations

import sys
from pathlib import Path

from e2e.errors import StepFailed

REPO = Path(__file__).resolve().parents[2]


def instance_for(label: str):
    """The registered ProvisionedInstance for `label` (its .target is the RemoteTarget)."""
    for path in (REPO / "skills" / "launch", REPO):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from launch_session import _read_config_instance
    inst = _read_config_instance(label)
    if inst is None:
        raise StepFailed(f"no instance under label {label!r} in provisioning/eval_config.json")
    return inst
