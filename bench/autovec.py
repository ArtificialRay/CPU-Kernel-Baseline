"""The autovectorized control: a dataset's scalar starting point, compiled with auto-vectorized sve/sve2/sme,
which exactly like an agent candidate.

`candidate_compile_flags` repeats the sessions' rule; bench/test_autovec.py compares it with each
session's `make_solution`, so a change on either side fails there.
"""

from __future__ import annotations

from typing import List, Sequence

from bench.data import Solution
from contracts import ISA_TABLE

AUTOVEC_AUTHOR = "autovec"


def candidate_compile_flags(dataset: str, start_flags: Sequence[str], march: str) -> List[str]:
    """The compile flags a session gives an agent's kernel for `dataset` on a tier with this `march`."""
    if dataset == "kleidiai":
        return ["-O3", march, "-std=c++17"]
    base = [f for f in start_flags if not f.startswith("-O") and not f.startswith("-march=")]
    return ["-O3", march, *base]


def autovec_build(start: Solution, isa: str) -> Solution:
    """`start` (a dataset's scalar starting point) as an agent candidate for tier `isa` would be built.

    Built through model_validate, not model_copy, so the solution hash, and with it the build cache
    key, is recomputed for the new flags."""
    spec = ISA_TABLE[isa]
    data = start.model_dump(mode="json")
    data["name"] = f"{AUTOVEC_AUTHOR}_{start.definition}"
    data["author"] = AUTOVEC_AUTHOR
    data["description"] = f"{start.name} compiled like an agent candidate for {isa}: compiler auto-vectorization only."
    data["spec"]["compile_flags"] = candidate_compile_flags(start.dataset.value, start.spec.compile_flags, spec.march)
    data["spec"]["isa_features"] = list(spec.features)
    data["spec"]["target_hardware"] = list(spec.labels)
    return Solution.model_validate(data)
