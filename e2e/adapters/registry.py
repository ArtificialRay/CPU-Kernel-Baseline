"""resolve_adapter(name): the e2e analogue of bench.compile.registry.BuilderRegistry."""
from __future__ import annotations

from typing import Callable

from .base import FrameworkAdapter

FACTORIES: dict[str, Callable[[], FrameworkAdapter]] = {}


def register_adapter(name: str, factory: Callable[[], FrameworkAdapter]) -> None:
    FACTORIES[name] = factory


def resolve_adapter(name: str) -> FrameworkAdapter:
    # Imported on demand: an adapter module pulls in its framework's dependencies.
    if name == "llamacpp" and name not in FACTORIES:
        from .llamacpp.adapter import LlamaCppAdapter
        register_adapter(name, LlamaCppAdapter)
    if name not in FACTORIES:
        raise SystemExit(f"no e2e adapter named {name!r}; known: {sorted({'llamacpp', *FACTORIES})}")
    return FACTORIES[name]()
