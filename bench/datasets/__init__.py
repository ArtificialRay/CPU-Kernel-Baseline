"""Per-dataset Python adapters: numpy ↔ framework-native tensor handles.

The C-side calling convention for each dataset lives with its builder, under
`bench/compile/builders/<dataset>_harness/<op_type>.{cpp,h}`. The Python-side
adapter here is responsible for:

  - `wrap_inputs(definition, workload, np_inputs)` — turn numpy arrays into
    opaque pointers the harness shim expects (ncnn::Mat*, raw float*, ...)
  - `unwrap_output(definition, workload, output_handle)` — read result back to numpy
  - `SIGNATURES` dict — kernel_type → ctypes argtypes list, mirrors
    `armbench_entry_<op_type>` declared in `<dataset>_harness/<op_type>.h`

To add a new dataset (e.g. "xnnpack"): create
bench/compile/builders/xnnpack_harness/ and add a new bench/datasets/xnnpack.py
with the same interface.
"""

from typing import Dict, Type

from .ncnn import NcnnDataset
from .raw import RawDataset

# Registry of dataset adapters by name. Keys must match SupportedDatasets enum.
DATASETS: Dict[str, Type] = {
    "ncnn": NcnnDataset,
    "raw": RawDataset,
}


def get(name: str):
    """Look up a dataset adapter class by name (e.g. 'ncnn')."""
    if name not in DATASETS:
        raise KeyError(
            f"Unknown dataset '{name}'. Registered datasets: {sorted(DATASETS)}"
        )
    return DATASETS[name]
