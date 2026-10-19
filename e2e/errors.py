"""A step that cannot continue. Library code raises it instead of exiting, so a caller
(the CLI, a test, another script) decides what a failure means."""


class StepFailed(RuntimeError):
    pass
