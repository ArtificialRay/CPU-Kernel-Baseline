"""NCNNKernelSession — KernelSession implementation for the NCNN dataset.

Adapted from eval/agent_tools/ncnn.py: agent writes `kernel.cpp` containing
`inner_<op_type>(...)`. The harness files (`<op_type>.h` + `<op_type>.cpp`) are
lifted from the existing reference-scalar solution for the definition, so the
agent only needs to write the inner kernel — not the binding boilerplate.

ISA/march/target_hardware now come from `isa.march_for_isa(self._isa)` (the
session's explicit, caller-supplied ISA) instead of an EC2-instance-type
table lookup — see mcp_app/agent_tools/isa.py.
"""

from __future__ import annotations

from typing import Optional

from bench.data.solution import Solution, SourceFile, SolutionSpec, SupportedDatasets

from . import isa
from .base import KernelSession, standard_tool_schemas


class NCNNKernelSession(KernelSession):
    """Tool surface for NCNN-dataset definitions (conv2d, etc.)."""

    dataset = "ncnn"

    def make_solution(self, code: str) -> Solution:
        """Wrap agent-written kernel.cpp into a Solution alongside harness files.

        Harness files (e.g. conv2d.h + conv2d.cpp) are lifted from the
        reference-scalar solution for this definition. compile_flags,
        isa_features, and target_hardware are derived from the session's
        explicit isa — always the same flags for a given isa value.
        """
        ref = self._trace_set.get_baseline_solution(
            self._definition.name, "reference-scalar"
        )
        if ref is None:
            raise ValueError(
                f"No reference-scalar solution for definition {self._definition.name!r} "
                "in TraceSet — cannot build NCNN candidate harness. "
                "Run scripts/gen_definitions.py to generate reference solutions."
            )

        harness = [s for s in ref.sources if s.path != "kernel.cpp"]
        if not harness:
            raise ValueError(
                f"reference-scalar solution for {self._definition.name!r} has no "
                "harness files besides kernel.cpp — unexpected layout"
            )

        agent_kernel = SourceFile(path="kernel.cpp", content=code)

        march, isa_features, target_hardware = isa.march_for_isa(
            self._isa, instance_label=self._instance_label
        )

        base_flags = [
            f for f in ref.spec.compile_flags
            if not f.startswith("-O") and not f.startswith("-march=")
        ]
        compile_flags = ["-O3", march, *base_flags]

        return Solution(
            name=f"{self._author}_{self._definition.name}",
            definition=self._definition.name,
            dataset=SupportedDatasets.NCNN,
            author=self._author,
            spec=SolutionSpec(
                language=ref.spec.language,
                target_hardware=target_hardware,
                entry_point=ref.spec.entry_point,
                dependencies=list(ref.spec.dependencies),
                isa_features=isa_features,
                compile_flags=compile_flags,
                link_flags=list(ref.spec.link_flags),
            ),
            sources=[*harness, agent_kernel],
        )

    def disassemble(self, fn: Optional[str] = None) -> dict:
        """Disassemble the agent's inner kernel (not the binding wrapper)."""
        if fn is None:
            fn = f"inner_{self._definition.op_type}"
        return super().disassemble(fn=fn)

    @classmethod
    def tool_schemas(cls) -> list[dict]:
        return standard_tool_schemas(
            code_description=(
                "Full C++ source for kernel.cpp. Must implement the "
                "`inner_<op_type>` function declared in `<op_type>.h` (the "
                "harness files are provided automatically). Use `extern \"C\"` "
                "if needed; the symbol does NOT need to be exported."
            ),
            disasm_hint="inner_<op_type>",
        )


__all__ = ["NCNNKernelSession"]
