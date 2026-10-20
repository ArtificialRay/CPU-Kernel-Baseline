"""The concrete axis values of a workload, as the harness derives them."""
from __future__ import annotations

from bench.data import Definition


def resolve_axes(definition: Definition, workload_axes: dict, scalars: dict | None = None, strict: bool = True) -> dict:
    """Const axes, the workload's axes, scalar inputs (`scalars`: name -> value), then the axes the definition's
    `constraints` derive (`H_out == H - Kh + 1` and the like). With strict=False a constraint that cannot be
    evaluated yet (a name it needs is unknown) is skipped instead of raising."""
    axes = {**definition.const_axes, **workload_axes}
    for name, value in (scalars or {}).items():
        axes[name] = int(value)
    for constraint in definition.constraints:
        lhs, _, rhs = constraint.partition("==")
        try:
            axes[lhs.strip()] = int(eval(rhs.strip(), {"__builtins__": {}}, axes))     # constraints are dataset content
        except NameError:
            if strict:
                raise
    return axes


def input_shape(definition: Definition, name: str, axes: dict) -> tuple | None:
    """The shape the definition declares for input `name` under `axes`; None when an axis is unknown."""
    spec = definition.inputs[name].shape
    if spec is None or any(a not in axes for a in spec):
        return None
    return tuple(axes[a] for a in spec)
