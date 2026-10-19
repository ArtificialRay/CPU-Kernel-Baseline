"""What one e2e run is about: which framework, which model, which ISA, and where its
artifacts live. Kept free of heavy imports so it loads on any box."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODELS_CONFIG = REPO / "config" / "e2e_models.json"
DEFAULT_FRAMEWORK = "llamacpp"


def load_models() -> dict:
    """config/e2e_models.json without its `_comment`-style keys."""
    raw = json.loads(MODELS_CONFIG.read_text())
    return {k: v for k, v in raw.items() if not k.startswith("_")}


@dataclass(frozen=True)
class RunDir:
    """File names inside one run directory, so no stage hard-codes them."""

    root: Path

    @property
    def gguf_head(self) -> Path:
        return self.root / "head.bin"

    @property
    def inventory(self) -> Path:
        return self.root / "inventory.json"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def results(self) -> Path:
        return self.root / "e2e_results.json"


@dataclass(frozen=True)
class E2EContext:
    framework: str
    model: str
    isa: str
    run: RunDir
    force: bool = False

    @classmethod
    def create(cls, framework: str, model: str, isa: str, run_dir: Path | None = None, force: bool = False) -> "E2EContext":
        root = run_dir or REPO / "e2e-runs" / f"{framework}-{model}-{isa}"
        return cls(framework=framework, model=model, isa=isa, run=RunDir(Path(root)), force=force)

    @property
    def model_config(self) -> dict:
        models = load_models()
        if self.model not in models:
            raise SystemExit(f"unknown model {self.model!r}; known: {sorted(models)}")
        return models[self.model]
