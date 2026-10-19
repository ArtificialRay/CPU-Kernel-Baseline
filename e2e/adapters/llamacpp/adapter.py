"""llama.cpp / ggml: every stage is one of the modules next to this file."""
from __future__ import annotations

import importlib
import sys
import urllib.request
from pathlib import Path

from ..base import Entrypoint, FrameworkAdapter

HEADER_BYTES = 64 * 1024 * 1024    # the GGUF header (metadata + tensor table) fits well inside this


def module_entrypoint(module: str) -> Entrypoint:
    """Run `<module>.main()` as if from the command line with `argv`.

    Imports lazily so listing stages never needs numpy/ml_dtypes, and sets sys.argv
    because the stage scripts parse their own arguments."""
    def run(argv: list[str]) -> int:
        mod = importlib.import_module(f"{__package__}.{module}")
        saved = sys.argv
        sys.argv = [f"e2e {module}", *argv]
        try:
            return int(mod.main() or 0)
        finally:
            sys.argv = saved
    return run


class LlamaCppAdapter(FrameworkAdapter):
    name = "llamacpp"

    def entrypoints(self) -> dict[str, Entrypoint]:
        return {stage: module_entrypoint(stage) for stage in (
            "inventory", "generate", "capture", "calibrate", "kernel_set", "reference_manifest",
            "verify", "deploy", "measure", "report", "report_from_log",
        )}

    def fetch_header(self, model_config: dict, dest: Path) -> Path:
        url = f"https://huggingface.co/{model_config['repo']}/resolve/main/{model_config['file']}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url, headers={"Range": f"bytes=0-{HEADER_BYTES - 1}"})
        with urllib.request.urlopen(request) as response, open(dest, "wb") as out:
            out.write(response.read())
        return dest
