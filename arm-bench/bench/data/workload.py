"""Concrete tensor sizes + scalar inputs for one benchmark point."""

import uuid as _uuid
from typing import Dict, Optional, Union

from pydantic import Field

from .utils import BaseModelWithDocstrings, NonEmptyString, PositiveInt


class Workload(BaseModelWithDocstrings):
    """One workload point: concrete values for a Definition's var axes + scalar inputs.

    Inputs/outputs sized from `definition.axes (const) ∪ self.axes (var)`. Tensor
    data is generated deterministically at run time from these axes (port of
    `make_mat_ramp` / `make_weights`); we don't store raw tensor blobs for Phase 1.

    `scalar_inputs` carries any non-tensor inputs declared in the Definition with
    `shape: null` (e.g. `activation_type: 0`, `pad_left: 1`).
    """

    axes: Dict[NonEmptyString, PositiveInt]
    """Concrete values for the Definition's var axes (e.g. {'N': 1, 'H': 56, 'W': 56})."""
    scalar_inputs: Dict[NonEmptyString, Union[int, float, bool]] = Field(default_factory=dict)
    uuid: NonEmptyString = Field(default_factory=lambda: _uuid.uuid4().hex)
    """Stable identifier so traces can be joined to workloads across runs."""
    tags: Dict[str, str] = Field(default_factory=dict)
    """Free-form metadata (e.g. {'from': 'test_conv_base_3x3_s1'})."""
    description: Optional[str] = None
