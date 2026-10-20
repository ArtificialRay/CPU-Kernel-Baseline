"""Where a framework's operators come from: its model. One module per framework, no shared interface yet.

Only frameworks (which deploy models) have a `discover`; for a library of kernels (simd-loop, kleidiai) the
agent states the operator directly and uses `kgen add`.
"""
