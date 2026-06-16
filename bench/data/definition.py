"""Framework-agnostic kernel Definition (axes, inputs, outputs, reference)."""

from __future__ import annotations

import ast
from enum import Enum
from functools import cached_property
from typing import Dict, List, Literal, Optional, Tuple, Union

from pydantic import Field, model_validator

from .utils import BaseModelWithDocstrings, NonEmptyString, NonNegativeInt


class ScratchBufSpec(BaseModelWithDocstrings):
    """A scratch buffer allocated by the harness for simd-loop kernels."""
    name: NonEmptyString
    dtype: NonEmptyString


class SimdLoopMeta(BaseModelWithDocstrings):
    """Adapter metadata for simd-loop definitions.

    These fields are harness implementation details (not part of the math spec)
    and are used by SimdLoopDataset.wrap_inputs to set up the calling convention.
    """
    output_inplace: bool = False
    """True when the kernel sorts/modifies its first input array in-place."""
    array_pad: int = Field(default=0, ge=0)
    """Extra elements allocated past N for kernels that read/write beyond the array."""
    scratch: list[ScratchBufSpec] = Field(default_factory=list)
    """Scratch buffers (name + dtype) passed between inputs and N in the ABI."""
    axes_order: list[str] = Field(default_factory=list)
    """Ordered axis names passed as int64 args in the ABI (multi-axis loops, e.g.
    ['m', 'n', 'k']). Empty → single-N loop (n derived from the first input)."""


class AxisConst(BaseModelWithDocstrings):
    """A dimension whose value is fixed across all workloads of this Definition."""

    type: Literal["const"] = "const"
    value: NonNegativeInt
    description: Optional[str] = None


class AxisVar(BaseModelWithDocstrings):
    """A dimension whose value varies per workload (e.g., batch size, H, W)."""

    type: Literal["var"] = "var"
    parent: Optional[str] = None
    """Name of a parent axis if this var is nested (e.g., per-batch sequence length)."""
    description: Optional[str] = None


AxisSpec = Union[AxisConst, AxisVar]


class DType(str, Enum):
    FLOAT64 = "float64"
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    INT64 = "int64"
    INT32 = "int32"
    INT16 = "int16"
    INT8 = "int8"
    UINT64 = "uint64"
    UINT32 = "uint32"
    UINT16 = "uint16"
    UINT8 = "uint8"
    BOOL = "bool"


class TensorSpec(BaseModelWithDocstrings):
    """A named tensor in a Definition's inputs or outputs."""

    shape: Optional[List[NonEmptyString]]
    """List of axis names. None means a scalar (passed as Python int/float/bool)."""
    dtype: DType
    description: Optional[str] = None


class Definition(BaseModelWithDocstrings):
    """Framework-agnostic specification of a kernel.

    Names follow the convention `<op_type>_<key>_<value>_<key>_<value>...`, e.g.
    `conv2d_kh3_kw3_sh1_sw1_dh1_dw1_c64_c128`. All constants live in `axes` (as
    `AxisConst`); only truly-per-workload dimensions are `AxisVar`.
    """

    name: NonEmptyString
    op_type: NonEmptyString
    """General compute category — drives `solutions/<dataset>/_harness/<op_type>.cpp` lookup."""
    axes: Dict[NonEmptyString, AxisSpec]
    inputs: Dict[NonEmptyString, TensorSpec]
    outputs: Dict[NonEmptyString, TensorSpec]
    reference: NonEmptyString
    """Python source. Must define a top-level `run(...)` whose kwarg names match `inputs`."""
    constraints: List[NonEmptyString] = Field(default_factory=list)
    tags: List[NonEmptyString] = Field(default_factory=list)
    description: Optional[str] = None
    simd_loop_meta: Optional[SimdLoopMeta] = None
    """Set for simd-loop definitions; drives SimdLoopDataset.wrap_inputs."""

    @model_validator(mode="after")
    def _validate_reference(self) -> "Definition":
        try:
            mod = ast.parse(self.reference, mode="exec")
        except SyntaxError as e:
            raise ValueError(f"Reference must be valid Python: {e}") from e
        if not any(isinstance(n, ast.FunctionDef) and n.name == "run" for n in mod.body):
            raise ValueError("Reference must define a top-level function named 'run'")
        return self

    @model_validator(mode="after")
    def _validate_input_output_names(self) -> "Definition":
        if set(self.inputs) & set(self.outputs):
            raise ValueError("Input and output names must not overlap")
        return self

    @model_validator(mode="after")
    def _validate_axis_references(self) -> "Definition":
        for name, spec in {**self.inputs, **self.outputs}.items():
            if spec.shape is None:
                continue
            for axis in spec.shape:
                if axis not in self.axes:
                    raise ValueError(f"Tensor '{name}' references undefined axis '{axis}'")
        return self

    @model_validator(mode="after")
    def _validate_constraints(self) -> "Definition":
        for c in self.constraints:
            try:
                ast.parse(c, mode="eval")
            except SyntaxError as e:
                raise ValueError(f"Constraint '{c}' is not a valid Python expression: {e}") from e
        return self

    @cached_property
    def const_axes(self) -> Dict[str, int]:
        return {n: a.value for n, a in self.axes.items() if isinstance(a, AxisConst)}

    @cached_property
    def var_axes(self) -> List[str]:
        return [n for n, a in self.axes.items() if isinstance(a, AxisVar)]

    def resolve_shape(
        self, tensor_name: str, workload_axes: Dict[str, int]
    ) -> Optional[Tuple[int, ...]]:
        """Resolve a tensor's symbolic shape given concrete values for var axes."""
        spec = self.inputs.get(tensor_name) or self.outputs.get(tensor_name)
        if spec is None:
            raise KeyError(f"Unknown tensor: {tensor_name}")
        if spec.shape is None:
            return None
        out: List[int] = []
        for axis_name in spec.shape:
            ax = self.axes[axis_name]
            if isinstance(ax, AxisConst):
                out.append(ax.value)
            else:
                if axis_name not in workload_axes:
                    raise ValueError(
                        f"Tensor '{tensor_name}' needs value for var axis '{axis_name}' "
                        f"but workload provides only {sorted(workload_axes)}"
                    )
                out.append(workload_axes[axis_name])
        return tuple(out)
