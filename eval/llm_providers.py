"""eval/llm_providers.py — optional per-provider api_key/api_base overrides
for Path 1's litellm agent loop (eval/evaluator.py::run_agentic_eval).

Without this file, litellm resolves credentials purely from provider-specific
env vars (ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY, ...), which
still works unchanged. eval/llm_providers.json (gitignored, copy from
.example) is opt-in: only providers/fields actually present there override
litellm's normal resolution, so a partially-filled or missing file falls back
to env vars for whatever's left out.
"""

import json
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent / "llm_providers.json"


def resolve_completion_kwargs(model: str) -> dict:
    """litellm.completion() kwarg overrides (api_key/api_base/...) for
    `model`'s provider prefix (e.g. "anthropic" for "anthropic/claude-opus-4-8"),
    read from eval/llm_providers.json. Empty dict if the file, the provider
    entry, or a given field is absent.
    """
    if not _CONFIG_PATH.exists():
        return {}
    providers = json.loads(_CONFIG_PATH.read_text()).get("providers", {})
    provider = providers.get(model.split("/")[0], {})
    return {k: v for k, v in provider.items() if v}
