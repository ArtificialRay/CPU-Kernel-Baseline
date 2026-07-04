"""SIMDLoopAgentTools — AgentTools implementation for the simd-loop dataset.

Agent writes `kernel.cpp` containing `inner_<op_type>(...)` (op_type is
`loop_NNN`). The fused harness files (`loop_NNN.h` + `loop_NNN.cpp`) are lifted
from the baseline reference solution for the definition, so the agent only needs
to write the inner loop kernel — not the harness boilerplate.

Same contract shape as the ncnn dataset (`inner_<op_type>` behind a fused
harness); the differences are the dataset enum, the baseline author to lift the
harness from ("reference", the scalar baseline), and dropping the reference's
`-fno-vectorize` flags so the candidate is actually allowed to vectorize.
"""

from __future__ import annotations

from typing import Optional

from bench.data.solution import (
    Solution,
    SourceFile,
    SolutionSpec,
    SupportedDatasets,
)

from .base import AgentTools, derive_isa, standard_tool_schemas


class SIMDLoopAgentTools(AgentTools):
    """Tool surface for simd-loop dataset definitions (loop_NNN)."""

    dataset = "simd-loop"

    @classmethod
    def can_handle(cls, dataset: str) -> bool:
        return dataset == "simd-loop"

    def make_solution(self, code: str) -> Solution:
        """Wrap agent-written kernel.cpp into a Solution alongside the fused harness.

        Harness files (loop_NNN.h + loop_NNN.cpp) are lifted from the baseline
        reference solution. compile_flags/isa_features/target_hardware are derived
        from the remote instance type; the reference's scalar-only flags
        (-fno-vectorize/-fno-slp-vectorize/-O*) are dropped so the candidate can
        vectorize with -O3 + the instance's SVE/SVE2 march.
        """
        ref_author = self._bench_cfg.baseline_author  # "reference" for simd-loop
        ref = self._trace_set.get_baseline_solution(self._definition.name, ref_author)
        if ref is None:
            raise ValueError(
                f"No '{ref_author}' baseline solution for definition "
                f"{self._definition.name!r} in TraceSet — cannot build simd-loop "
                "candidate harness. Run scripts/gen_simd_loop_harness.py to generate it."
            )

        harness = [s for s in ref.sources if s.path != "kernel.cpp"]
        if not harness:
            raise ValueError(
                f"'{ref_author}' solution for {self._definition.name!r} has no harness "
                "files besides kernel.cpp — unexpected layout"
            )

        agent_kernel = SourceFile(path="kernel.cpp", content=code)

        march, isa_features, target_hardware = derive_isa(self._handle.instance_type)

        # Inherit non-optimization, non-march, non-scalar-forcing flags from the
        # reference (e.g. -std=c++14), then apply -O3 + march so the agent can vectorize.
        base_flags = [
            f for f in ref.spec.compile_flags
            if not f.startswith("-O")
            and not f.startswith("-march=")
            and f not in ("-fno-vectorize", "-fno-slp-vectorize")
        ]
        compile_flags = ["-O3", march, *base_flags]

        return Solution(
            name=f"{self._author}_{self._definition.name}",
            definition=self._definition.name,
            dataset=SupportedDatasets.SIMD_LOOP,
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
        """Disassemble the agent's inner kernel (not the harness wrapper)."""
        if fn is None:
            fn = f"inner_{self._definition.op_type}"
        return super().disassemble(fn=fn)

    @classmethod
    def tool_schemas(cls) -> list[dict]:
        return standard_tool_schemas(
            code_description=(
                "Full C++ source for kernel.cpp. Must implement the "
                "`inner_<op_type>` function declared in `<op_type>.h` (e.g. "
                "`extern \"C\" void inner_loop_001(struct loop_001_data *data)`). "
                "The harness calls it internally; the symbol need not be exported."
            ),
            disasm_hint="inner_<op_type>",
        )


__all__ = ["SIMDLoopAgentTools"]
