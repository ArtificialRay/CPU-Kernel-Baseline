"""Turn the activations of a real model into workload inputs, so the correctness of a quantized operator is
judged on realistic data instead of random numbers.

Why: random activations are near-Gaussian and zero-mean. A real post-norm hidden state has a few channels one to
two orders of magnitude larger than the rest and block sums far from zero. Numerical shortcuts that cost almost
nothing on random data (one activation scale per row instead of per block, an int8 requantization of block sums
under one exponent shared across a batch) cost 3.9 perplexity in a real model; only real inputs expose them.

How: the model is run in PyTorch with a forward hook on the layers that match a definition. Which model, which
weights, which data to feed and which layers count as an operator input are the caller's business and live in a
small loader file, written when it is needed:

    # load.py
    from kgen.capture import Adapter

    def load():
        model = ...                      # a torch.nn.Module in eval mode
        batches = [...]                  # each item: a tensor / tuple (positional) or a dict (keyword) for model(...)
        return model, batches

    def adapters():                      # how layers of this model show up as definition inputs
        return [Adapter(
            kind="linear",
            match=lambda layer: type(layer).__name__ == "Linear",
            params=lambda layer: {"K": layer.in_features, "N": layer.out_features},   # what a definition's const axes must equal
            named=lambda layer, x: (x.reshape(-1, x.shape[-1]), ("M", "K")),          # input as (tensor, axis names), or None to skip
            axes=("M", "K"),             # the axis names `named` returns
            slice_axis="M")]             # the axis a workload may take a window of

    python -m kgen capture pytorch --loader load.py --source "Qwen3-4B / wikitext-2 chunk 0" --tag e2e:qwen3.5-4b

kgen knows no layer type: Linear, Conv2d, a norm, an MoE block, an attention module are all adapters the loader
defines (the same ones feed kgen.collect). A layer matches a definition when the definition's const axes agree
with the adapter's `params` (two of them, or all when the definition has fewer) and the definition's input has
exactly the adapter's `axes`. The input is put into the axis order and dtype the definition declares, saved by
kgen.tensors, and the workload's input becomes a `tensor` input (the workload keeps its uuid). Workloads with no
suitable sample (a spatial size the model never produced) stay as they were; run the model at another resolution
to cover them.
"""
from __future__ import annotations

import importlib.util
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from bench.data import Definition

from .axes import input_shape, resolve_axes
from .errors import KgenError
from .layout import workload_path
from .tensors import attach_inputs

MIN_COMMON_PARAMS = 2     # a definition and a layer must agree on at least this many named constants


@dataclass(frozen=True)
class Adapter:
    """How one kind of layer shows up as a definition input; defined by the loader file, not by kgen."""
    kind: str
    match: Callable                             # layer -> bool: is this layer an instance of the operator
    params: Callable                            # layer -> {axis name: value} that a definition's const axes must match
    named: Callable                             # (layer, input tensor) -> (tensor, axis names of its dims), or None
    axes: tuple                                 # the axis names `named` returns, known before any call
    slice_axis: str                             # axis a workload may take a leading/windowed part of


@dataclass
class Sample:
    layer_name: str
    tensor: object                              # torch.Tensor, float32, on cpu
    axes: tuple


@dataclass
class Target:
    definition: Definition
    input_name: str
    adapter: Adapter
    params: dict
    samples: list = field(default_factory=list)      # capture: Sample per recorded call
    shapes: list = field(default_factory=list)       # collect: {axis name: size} per recorded call


def agrees(definition: Definition, input_name: str, adapter: Adapter, params: dict, axes: tuple) -> bool:
    """The definition's const axes agree with the layer's parameters (and enough of them are named: two, or all
    of them when the definition has fewer), and the definition's input has exactly the axes the layer's input has."""
    shape = definition.inputs[input_name].shape
    if shape is None or set(shape) != set(axes):
        return False
    named = [a for a in params if a in definition.axes]
    return (len(named) >= min(MIN_COMMON_PARAMS, len(definition.const_axes))
            and all(definition.const_axes[a] == params[a] for a in named if a in definition.const_axes))


def load_loader(path: Path):
    """(model, batches, adapters) from a loader file that defines load() and adapters()."""
    spec = importlib.util.spec_from_file_location("kgen_capture_loader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, what in (("load", "load() -> (model, batches)"), ("adapters", "adapters() -> [kgen.capture.Adapter, ...]")):
        if not hasattr(module, name):
            raise KgenError(f"{path} must define {what}; see the docstring of kgen/capture.py")
    adapters = list(module.adapters())
    bad = [a for a in adapters if not isinstance(a, Adapter)]
    if bad or not adapters:
        raise KgenError(f"{path}: adapters() must return a non-empty list of kgen.capture.Adapter, got {adapters!r}")
    model, batches = module.load()
    return model, batches, adapters


def spread(items: list, count: int) -> list:
    """`count` items spread evenly over the list (all of them when it is shorter)."""
    if len(items) <= count:
        return items
    return [items[round(i * (len(items) - 1) / (count - 1))] for i in range(count)] if count > 1 else items[:1]


def run_hooked(model, batches, adapters: list, targets_for: Callable, modules_per_shape: int,
               calls_per_module: int | None, on_call: Callable) -> None:
    """Run the model once over `batches` with a forward pre-hook on the layers that match; `targets_for(adapter,
    layer)` lists the targets a layer can feed. A layer belongs to the first adapter whose `match` accepts it.
    Each recorded call (at most `calls_per_module` per layer, all of them when None) is handed to
    `on_call(layer name, adapter, targets, tensor, axis names)`: the layer's input as the adapter names it, still
    in the model's dtype and device."""
    import torch

    groups: dict = {}
    for name, layer in model.named_modules():
        try:
            adapter = next((a for a in adapters if a.match(layer)), None)
            if adapter is None:
                continue
            targets = targets_for(adapter, layer)
            key = (adapter.kind, tuple(sorted(adapter.params(layer).items())))
        except Exception as e:      # the adapter is the loader's code: say which one failed on which layer
            raise KgenError(f"an adapter failed on layer {name!r} ({type(layer).__name__}): {type(e).__name__}: {e}") from e
        if targets:
            groups.setdefault(key, []).append((name, layer, adapter, targets))

    handles, seen = [], {}
    for members in groups.values():
        for name, layer, adapter, targets in spread(members, modules_per_shape):
            def hook(module, args, name=name, adapter=adapter, targets=targets):
                if (calls_per_module is not None and seen.get(name, 0) >= calls_per_module) or not args:
                    return
                named = adapter.named(module, args[0].detach())
                if named is None:
                    return
                seen[name] = seen.get(name, 0) + 1
                on_call(name, adapter, targets, *named)
            handles.append(layer.register_forward_pre_hook(hook))
    try:
        with torch.no_grad():
            for batch in batches:
                if isinstance(batch, dict):
                    model(**batch)
                elif isinstance(batch, (tuple, list)):
                    model(*batch)
                else:
                    model(batch)
    finally:
        for h in handles:
            h.remove()


def pick(target: Target, row: dict, index: int):
    """The array for one workload row from the target's samples, in the definition's axis order; None when no
    sample fits."""
    definition, input_name = target.definition, target.input_name
    shape = definition.inputs[input_name].shape
    want = input_shape(definition, input_name, resolve_axes(definition, row["axes"], strict=False))
    if want is None:
        return None
    wanted = dict(zip(shape, want))
    for k in range(len(target.samples)):
        sample = target.samples[(index + k) % len(target.samples)]
        dims = dict(zip(sample.axes, sample.tensor.shape))
        if any(dims[a] != wanted[a] for a in dims if a != target.adapter.slice_axis):
            continue
        take = wanted[target.adapter.slice_axis]
        if dims[target.adapter.slice_axis] < take:
            continue
        start = (index * 37) % max(1, dims[target.adapter.slice_axis] - take + 1)
        axis = sample.axes.index(target.adapter.slice_axis)
        part = sample.tensor.narrow(axis, start, take)
        part = part.permute(*[sample.axes.index(a) for a in shape]).contiguous().numpy()
        label = f"{re.sub(r'[^A-Za-z0-9]+', '-', sample.layer_name).strip('-')}_r{start}"
        return part, label, f"{sample.layer_name} rows {start}:{start + take}"
    return None


def select_targets(root: Path, definitions=None, tag=None, input_name: str | None = None) -> list:
    """(definition, input name) of every selected definition: the named ones and those carrying `tag`. The input is
    `input_name`, or the definition's first."""
    selected = {}
    for path in sorted((Path(root) / "definitions").glob("*/*.json")):
        d = Definition.model_validate_json(path.read_text())
        if d.name in (definitions or []) or (tag and tag in d.tags):
            selected[d.name] = d
    missing = set(definitions or []) - set(selected)
    if missing:
        raise KgenError(f"no such definition(s): {sorted(missing)}")
    if not selected:
        raise KgenError("nothing selected (pass --definitions or --tag)")

    targets = []
    for d in selected.values():
        name = input_name or next(iter(d.inputs))
        if name not in d.inputs:
            raise KgenError(f"{d.name} has no input {name!r}; inputs are {list(d.inputs)}")
        targets.append((d, name))
    return targets


def make_targets_for(targets: list, bound: dict) -> Callable:
    """`targets_for` of run_hooked: the Target of every selected definition a layer can feed, one Target per
    (definition, kind of layer, layer parameters), kept in `bound`."""
    def targets_for(adapter, layer):
        params = adapter.params(layer)
        found = []
        for d, n in targets:
            if agrees(d, n, adapter, params, adapter.axes):
                found.append(bound.setdefault((d.name, n, adapter.kind, tuple(sorted(params.items()))),
                                              Target(d, n, adapter, params)))
        return found
    return targets_for


def report_unmatched(targets: list, bound: dict) -> None:
    for d, _ in targets:
        if not any(t.definition.name == d.name for t in bound.values()):
            print(f"{d.name}: no layer of the model matches this definition (its const axes / input axes)")


def capture_pytorch(root: Path, loader: Path, source: str, definitions=None, tag=None, input_name: str | None = None,
                    modules_per_shape: int = 4, calls_per_module: int = 2, max_rows: int = 4096,
                    dry_run: bool = False) -> int:
    """Rewrite one input of the selected definitions' workloads to activations captured from the model `loader`
    builds. Returns how many workload inputs were rewritten."""
    import torch

    root = Path(root)
    targets = select_targets(root, definitions, tag, input_name)
    bound: dict = {}

    def keep(name, adapter, found, tensor, axes):
        tensor = tensor.to(torch.float32).cpu()
        if axes[0] == adapter.slice_axis:
            tensor = tensor[:max_rows]
        for target in found:
            target.samples.append(Sample(name, tensor.clone(), axes))

    model, batches, adapters = load_loader(loader)
    run_hooked(model, batches, adapters, make_targets_for(targets, bound), modules_per_shape, calls_per_module, keep)
    report_unmatched(targets, bound)

    written = 0
    for target in bound.values():
        d = target.definition
        if not target.samples:
            print(f"{d.name}: no layer of the model produced a sample")
            continue
        path = workload_path(root, d)
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        items, skipped = [], 0
        for i, row in enumerate(rows):
            got = pick(target, row, i)
            if got is None:
                skipped += 1
                continue
            array, label, where = got
            items.append((i, array, label, where))
        if items:
            attach_inputs(root, d, target.input_name, items, source, dry_run)
        print(f"{d.name}: {len(items)}/{len(rows)} workloads -> real {target.input_name} "
              f"from {len(target.samples)} captured sample(s)" + (f", {skipped} left as they were" if skipped else ""))
        written += len(items)
    return written
