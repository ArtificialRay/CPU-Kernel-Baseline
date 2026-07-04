"""Concrete C++/AOT kernel Solution for an arm-bench Definition.

Differs from flashinfer-bench's Solution by adding CPU/AOT specifics:
- top-level `dataset` field (pins calling convention, must match parent folder)
- spec.isa_features (host-skip check)
- spec.compile_flags / spec.link_flags (clang++ AOT specifics)

Solution does NOT carry per-binding shim code. The shared C harness lives on
disk under `solutions/<dataset>/_harness/<op_type>.{cpp,h}` and is concatenated
with `sources` at compile time.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional

from pydantic import ConfigDict, Field, PrivateAttr, model_validator

from .utils import BaseModelWithDocstrings, NonEmptyString


class SupportedLanguages(str, Enum):
    CPP = "cpp"
    """Pure C++ source compiled to a shared library via clang++."""
    # Future: ASM, RUST, etc.


class SupportedDatasets(str, Enum):
    NCNN = "ncnn"
    """Calls the kernel via ncnn::Mat/Option scaffolding; harness wraps ncnn::Convolution etc."""
    RAW = "raw"
    """Calls the kernel with bare `float*` tensors; for non-framework solutions."""
    SIMD_LOOP = "simd-loop"
    """The loop_* SIMD baseline dataset. Uses SimdLoopBuilder + SimdLoopDataset."""
    LLAMA_CPP = "llama.cpp"
    """Calls the kernel via ggml tensors/graphs linked against llama.cpp's static
    libs. Uses LlamaCppBuilder + LlamaCppDataset."""
    # Future: XNNPACK, EIGEN, ...


class SourceFile(BaseModelWithDocstrings):
    """A single source file in a Solution. The benchmarker materializes these to a
    temp dir and runs clang++ on them together with the dataset's shared harness.
    """

    path: NonEmptyString
    """Relative path. No absolute paths, no '..' traversal."""
    content: NonEmptyString

    @model_validator(mode="after")
    def _validate_path(self) -> "SourceFile":
        p = Path(self.path)
        if p.is_absolute():
            raise ValueError(f"Source path must be relative: {self.path}")
        if ".." in p.parts:
            raise ValueError(f"Source path must not traverse parents: {self.path}")
        return self


class SolutionSpec(BaseModelWithDocstrings):
    """Build & calling specification for a Solution."""

    language: SupportedLanguages = SupportedLanguages.CPP
    target_hardware: List[NonEmptyString] = Field(min_length=1)
    """E.g. ['graviton3', 'aarch64-sve']. Informational; benchmarker may use for skip checks."""
    entry_point: NonEmptyString
    """`<source_file>::<C_symbol>`. The symbol the builder's harness shim
    `armbench_entry_<op_type>` calls into. Must follow the per-(dataset, op_type)
    contract declared in the builder's harness header (e.g.
    `bench/compile/builders/ncnn_harness/<op_type>.h` for ncnn baselines,
    `candidate_harness/<op_type>.h` for candidates)."""
    dependencies: List[NonEmptyString] = Field(default_factory=list)
    """E.g. ['ncnn', 'openmp']. Mapped to -l flags by the compile builders."""
    isa_features: List[NonEmptyString] = Field(default_factory=list)
    """E.g. ['sve', 'neon']. Benchmarker skips solutions whose features the host lacks."""
    compile_flags: List[NonEmptyString] = Field(default_factory=list)
    """Passed verbatim to clang++ (e.g. '-O3', '-march=armv8.2-a+sve', '-fopenmp')."""
    link_flags: List[NonEmptyString] = Field(default_factory=list)
    """Passed verbatim at link time (e.g. '-lncnn', '-lgomp')."""

    @model_validator(mode="after")
    def _validate_entry_point(self) -> "SolutionSpec":
        if self.entry_point.count("::") != 1:
            raise ValueError(
                f"entry_point must be '<file>::<symbol>', got: {self.entry_point}"
            )
        return self


class Solution(BaseModelWithDocstrings):
    """A concrete kernel implementation. Frozen/immutable so we can memoize the hash."""

    model_config = ConfigDict(use_attribute_docstrings=True, frozen=True)

    _hash_cache: str = PrivateAttr()

    name: NonEmptyString
    definition: NonEmptyString
    """Name of the Definition this implements."""
    dataset: SupportedDatasets
    """Calling convention. Must match parent folder under `solutions/`."""
    author: NonEmptyString
    spec: SolutionSpec
    sources: List[SourceFile] = Field(min_length=1)
    description: Optional[str] = None

    @model_validator(mode="after")
    def _validate_sources(self) -> "Solution":
        seen = set()
        for s in self.sources:
            if s.path in seen:
                raise ValueError(f"Duplicate source path: {s.path}")
            seen.add(s.path)
        entry_file = self.spec.entry_point.split("::")[0]
        if entry_file not in seen:
            raise ValueError(f"entry_point file '{entry_file}' not present in sources")
        return self

    def get_entry_file(self) -> str:
        return self.spec.entry_point.split("::")[0]

    def get_entry_symbol(self) -> str:
        return self.spec.entry_point.split("::")[1]

    def model_post_init(self, __ctx: Any) -> None:
        object.__setattr__(self, "_hash_cache", self._compute_hash())

    def _compute_hash(self) -> str:
        h = hashlib.sha1()
        for chunk in (
            self.definition,
            self.dataset.value,
            self.spec.language.value,
            self.spec.entry_point,
            *self.spec.dependencies,
            *self.spec.compile_flags,
            *self.spec.link_flags,
            *(p for s in self.sources for p in (s.path, s.content)),
        ):
            h.update(chunk.encode())
        return h.hexdigest()

    def hash(self) -> str:
        return self._hash_cache

    def __hash__(self) -> int:
        return hash(self._hash_cache)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Solution):
            return NotImplemented
        return self._hash_cache == other._hash_cache
