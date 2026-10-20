"""A kgen operation that cannot continue. Library code raises it instead of exiting, so the CLI (or the
e2e step loop, or a test) decides what a failure means."""


class KgenError(RuntimeError):
    pass
