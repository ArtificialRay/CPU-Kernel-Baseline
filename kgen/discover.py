"""Where a framework's operators come from: its model. kgen does not know any framework, so the way to read one is a
plugin file, written when it is needed:

    # plugin.py
    def discover(argv: list[str]) -> dict:
        ...                              # read a GGUF header, trace a torchvision model, ... -> a JSON-serializable dict

    python -m kgen discover plugin.py [--out inventory.json] [-- plugin arguments]

kgen loads the file, calls `discover` with the arguments after the plugin, checks that the result is a JSON object
and writes it to --out (or prints it). What the object holds (shapes, counts, byte shares, measured time) is the
plugin's and its consumers' business. Only frameworks, which deploy models, have a discover; for a library of
kernels (simd-loop, kleidiai) the agent states the operator directly and uses `kgen add`.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from .errors import KgenError


def run_plugin(plugin: Path, argv: list[str]) -> dict:
    spec = importlib.util.spec_from_file_location("kgen_discover_plugin", plugin)
    if spec is None or spec.loader is None:
        raise KgenError(f"{plugin} is not a python file")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "discover"):
        raise KgenError(f"{plugin} must define discover(argv: list[str]) -> dict")
    result = module.discover(list(argv))
    if not isinstance(result, dict):
        raise KgenError(f"{plugin}: discover() must return a dict, got {type(result).__name__}")
    try:
        json.dumps(result)
    except TypeError as e:
        raise KgenError(f"{plugin}: the dict discover() returned is not JSON-serializable ({e})")
    return result
