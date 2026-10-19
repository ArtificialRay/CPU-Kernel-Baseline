"""The framework-specific half of the e2e pipeline.

An adapter maps each stage name to the callable that does that stage's work for its
framework. The callable takes the stage's own command-line arguments (exactly what the
old scripts took) and returns an exit code, so `python -m e2e <stage> ...` is the same
program as before and the composite `prepare` / `evaluate` commands only have to build
the arguments.

The interface is deliberately this coarse: llama.cpp is the only framework so far, and
a finer split (inventory / emit_definitions / bench / quality ...) drawn from one
implementation would be a guess. Split it when a second framework shows what differs.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

Entrypoint = Callable[[list[str]], int]

STAGE_NAMES = (
    "inventory", "generate", "capture", "calibrate",                  # prepare: once per model
    "kernel_set", "deploy", "measure", "report", "report_from_log",
)


class FrameworkAdapter(ABC):
    name: str

    @abstractmethod
    def entrypoints(self) -> dict[str, Entrypoint]:
        """stage name -> callable(argv) -> exit code, for the stages this framework has."""

    @abstractmethod
    def fetch_header(self, model_config: dict, dest: Path) -> Path:
        """Download just enough of the model file for the `inventory` stage."""

    def entrypoint(self, stage: str) -> Entrypoint:
        table = self.entrypoints()
        if stage not in table:
            raise SystemExit(f"the {self.name!r} adapter has no {stage!r} stage (has: {sorted(table)})")
        return table[stage]
