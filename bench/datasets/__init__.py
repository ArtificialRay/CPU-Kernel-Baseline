"""Per-dataset Python adapters: numpy ↔ framework-native tensor handles.

The C-side `armbench_entry_<op_type>` binding ships in each solution's own
sources. The Python-side adapter here provides:

  - `wrap_inputs(np_inputs, op_type, lib, *, definition, out_shape)` — turn
    numpy arrays into the ctypes args the entry expects (ncnn::Mat*, raw float*, ...)
  - `unwrap_output(ctx)` — read the result back to numpy
  - `release(ctx)` — free anything wrap_inputs allocated

To add a dataset, add a bench/datasets/<name>.py with the same interface and
register it in DATASETS.
"""

from typing import Dict, Type

from .llama_cpp import LlamaCppDataset
from .ncnn import NcnnDataset
from .raw import RawDataset
from .simd_loop import SimdLoopDataset

# Registry of dataset adapters by name. Keys must match SupportedDatasets enum.
DATASETS: Dict[str, Type] = {
    "llama.cpp": LlamaCppDataset,
    "ncnn": NcnnDataset,
    "raw": RawDataset,
    "simd-loop": SimdLoopDataset,
}


def get(name: str):
    """Look up a dataset adapter class by name (e.g. 'ncnn')."""
    if name not in DATASETS:
        raise KeyError(
            f"Unknown dataset '{name}'. Registered datasets: {sorted(DATASETS)}"
        )
    return DATASETS[name]
