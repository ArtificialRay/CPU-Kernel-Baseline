"""KleidiaiKernelSession — KernelSession implementation for the kleidiai dataset.

Agent writes `kernel.cpp` containing `inner_<op_type>(...)` (e.g. `inner_gemm`
for the gemm op_type) behind the same raw `float*` ABI as reference-scalar.
The two harness files (`<op_type>.h` + `<op_type>.cpp`, e.g. `gemm.h` +
`gemm.cpp`) are lifted from the `reference-scalar` solution for the
definition — unlike simd-loop, these two files never change per-definition
in content shape (just the baked N/K-style constants), so there's no
per-definition harness generator involved; reference-scalar's own copy
already has the exact files a candidate needs.

Deliberately NOT lifted from `self._bench_cfg.baseline_author` (that's
`baseline-kleidiai-arm`, the real KleidiAI kernel — its sources are vendored
kai_*.c/.S files, not a gemm.h/gemm.cpp shim, and are useless as a candidate
harness).
"""

from __future__ import annotations

from typing import Optional

from bench.data.solution import Solution, SourceFile, SolutionSpec, SupportedDatasets
from contracts import AGENT_KERNEL_FILENAME, REFERENCE_SCALAR_AUTHORS

from . import isa
from .base import KernelSession
from .schemas import standard_tool_schemas


class KleidiaiKernelSession(KernelSession):
    """Tool surface for kleidiai dataset definitions (raw float* ABI)."""

    dataset = "kleidiai"

    def make_solution(self, code: str) -> Solution:
        """Wrap agent-written kernel.cpp into a Solution alongside the
        reference-scalar harness (<op_type>.h + <op_type>.cpp).
        """
        ref_author = REFERENCE_SCALAR_AUTHORS[self.dataset]
        ref = self._trace_set.get_baseline_solution(self._definition.name, ref_author)
        if ref is None:
            raise ValueError(
                f"No '{ref_author}' solution for definition {self._definition.name!r} "
                "in TraceSet — cannot build kleidiai candidate harness."
            )

        harness = [s for s in ref.sources if s.path != AGENT_KERNEL_FILENAME]
        if not harness:
            raise ValueError(
                f"'{ref_author}' solution for {self._definition.name!r} has no harness "
                f"files besides {AGENT_KERNEL_FILENAME} — unexpected layout"
            )

        agent_kernel = SourceFile(path=AGENT_KERNEL_FILENAME, content=code)

        march, isa_features, target_hardware = isa.march_for_isa(
            self._isa, instance_label=self._instance_label
        )

        return Solution(
            name=f"{self._author}_{self._definition.name}",
            definition=self._definition.name,
            dataset=SupportedDatasets.KLEIDIAI,
            author=self._author,
            spec=SolutionSpec(
                language=ref.spec.language,
                target_hardware=target_hardware,
                entry_point=ref.spec.entry_point,
                dependencies=list(ref.spec.dependencies),
                isa_features=isa_features,
                compile_flags=["-O3", march, "-std=c++17"],
                link_flags=list(ref.spec.link_flags),
            ),
            sources=[*harness, agent_kernel],
        )

    def disassemble(self, definition: str, version: int, fn: Optional[str] = None) -> dict:
        """Disassemble the agent's inner kernel (not the armbench_entry_<op> shim)."""
        if fn is None:
            fn = f"inner_{self._definition.op_type}"
        return super().disassemble(definition=definition, version=version, fn=fn)

    @classmethod
    def tool_schemas(cls) -> list[dict]:
        return standard_tool_schemas()


__all__ = ["KleidiaiKernelSession"]
