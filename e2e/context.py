"""What one e2e run is about: which framework, which model, which ISA, where its artifacts
live, and the options of the steps. Kept free of heavy imports so it loads on any box."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
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
    """File names inside one run directory, so no step hard-codes them."""

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


@dataclass
class Options:
    """Everything the steps take beyond the context. Flat on purpose: each field is one
    command-line flag of `prepare` / `evaluate`, and a step reads only the ones it needs."""

    steps: list[str] | None = None        # None: every local step; the box-side ones are only printed
    root: Path = REPO / "bench-trace"     # the warehouse the steps read and write

    # prepare
    gguf_head: str | None = None          # a local GGUF header; downloaded when omitted
    min_share: float = 0.001              # generate: skip shapes below this per-token byte share
    dump: str | None = None               # capture: ARMBENCH_DUMP_DIR contents
    source: str | None = None             # capture: provenance recorded on each real-activation input

    # evaluate: kernel_set
    author: str | None = None             # whose bench-trace/solutions/<dataset>/<author>/ to compile
    rows_abi: bool = False
    only: list[str] | None = None
    only_types: list[str] | None = None
    only_roles: list[str] | None = None
    reference_author: str | None = None   # guards: compare against this author's solutions
    expect_count: int | None = None
    expect_changed: list[str] = field(default_factory=list)
    expect_same: list[str] = field(default_factory=list)
    forbid: list[str] = field(default_factory=list)
    cxx: str = "clang++-18"
    march: str | None = None

    # evaluate: deploy
    label: str | None = None              # provisioning/eval_config.json instance label
    agent_build: bool = False

    # evaluate: measure (runs on the box) and report
    llama_dir: str = "~/llama.cpp-e2e"    # holds build-stock / build-norepack / build-agent
    threads: list[int] = field(default_factory=lambda: [4])
    pp: int = 512
    tg: int = 128
    reps: int = 5
    fa: str = "off"
    ub: int | None = None
    perplexity: str | None = None         # text file for llama-perplexity; omitted: no quality check
    ppl_chunks: int = 64
    baseline: str = "stock"
