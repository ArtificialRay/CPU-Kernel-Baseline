"""The Graviton box a step drives over SSH, looked up through provisioning/ by its label
(`e2e-<model>-<isa>` unless `--label` is given)."""
from __future__ import annotations

from e2e.errors import StepFailed


def get_box(label: str, *, create_isa: str | None = None):
    """The InstanceHandle (provisioning/remote.py) of the box under `label`.

    If there is none and `create_isa` is given, one for that ISA's instance type is provisioned first
    (with the llama.cpp-e2e tree built); otherwise a missing box is an error."""
    from provisioning.provision import get_running_instance, provision
    box = get_running_instance(label)
    if box is not None:
        return box
    if create_isa is None:
        raise StepFailed(
            f"no instance under label {label!r} in provisioning/eval_config.json; start one with "
            f"`python -m e2e evaluate --steps deploy --provision --label {label} ...` "
            f"or provisioning/provision.py --label {label} --isa <isa>")
    import contracts
    instance_type = contracts.ISA_INSTANCE_MAP[create_isa]
    print(f"[e2e] provisioning {instance_type} under label {label!r}")
    return provision(label, instance_type, "llama.cpp-e2e")
