"""A recipe is how one family of kernels gets its bench-trace material.

The pipeline has three stages, always in this order: `define` (Definition JSONs), `workload`
(workload JSONLs) and `solutions` (the reference and baseline Solution JSONs). A recipe supplies
`plan`, which reads the recipe's own arguments once into whatever the stages need, and one
function per stage, each `stage(plan, args) -> number of files written`. Stages overwrite what
they write.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

STAGES = ("define", "workload", "solutions")


@dataclass(frozen=True)
class Recipe:
    name: str
    description: str
    add_arguments: Callable[[Any], None]          # adds the recipe's options to an argparse parser
    plan: Callable[[Any], Any]                    # args -> plan shared by the stages
    define: Callable[[Any, Any], int]
    workload: Callable[[Any, Any], int]
    solutions: Callable[[Any, Any], int]


def registry() -> dict[str, Recipe]:
    from .llamacpp import RECIPE as llamacpp_gemm
    return {llamacpp_gemm.name: llamacpp_gemm}
