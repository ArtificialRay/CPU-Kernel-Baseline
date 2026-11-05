"""The Graviton box a step drives over SSH, through provisioning/.

e2e keeps one box per (model, isa) under the label `e2e-<model>-<isa>` (or the label given on the
command line). provisioning/provision.py records every instance in provisioning/eval_config.json;
this only asks it for the one under the label. A box is created only when `create` is true, which the
command line sets with `--provision` -- nothing here spends money otherwise."""
from __future__ import annotations

from e2e.errors import StepFailed


def get_box(label: str, *, create_isa: str | None = None):
    """The InstanceHandle (provisioning/remote.py) of the box under `label`.

    If there is none and `create_isa` is given, one for that ISA's instance type is provisioned first
    (spot, with the llama.cpp-e2e tree built). Otherwise a missing box is an error naming the command."""
    # Imported here: provisioning pulls in terraform/dotenv helpers that a local step never needs.
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
