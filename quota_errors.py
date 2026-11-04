"""Provider-neutral signals for exhausted account quota."""

from __future__ import annotations

import json
import re
from pathlib import Path

QUOTA_EXIT_CODE = 75

_HARD_QUOTA_CODES = {
    "usage_limit_reached",
    "usage_limit_exceeded",
    "model_requires_usage_credits",
    "insufficient_quota",
    "billing_hard_limit_reached",
    "credit_balance_exhausted",
    "project_spend_limit_exceeded",
    "organization_spend_limit_exceeded",
    "organization_usage_limit_exceeded",
}
_HARD_QUOTA_CODE_TEXT = re.compile(
    r"(?<![a-z_])(?:" + "|".join(sorted(_HARD_QUOTA_CODES)) + r")(?![a-z_])",
    re.IGNORECASE,
)
_HARD_QUOTA_TEXT = re.compile(
    r"\b(?:usage limit reached|billing hard limit reached|"
    r"credit balance is too low|insufficient credits|out of credits|"
    r"credits exhausted|account quota exhausted|monthly spending limit reached)\b",
    re.IGNORECASE,
)


def is_hard_quota_error(error: object) -> bool:
    """Require an explicit account-limit signal; HTTP 429 alone is transient."""
    if isinstance(error, BaseException):
        if is_hard_quota_error(getattr(error, "code", None)):
            return True
        if is_hard_quota_error(getattr(error, "body", None)):
            return True
        return is_hard_quota_error(str(error))
    if isinstance(error, dict):
        for key in ("code", "type", "api_error", "error", "message", "detail"):
            if is_hard_quota_error(error.get(key)):
                return True
        return False
    if isinstance(error, str):
        return (error.lower() in _HARD_QUOTA_CODES
                or bool(_HARD_QUOTA_CODE_TEXT.search(error))
                or bool(_HARD_QUOTA_TEXT.search(error)))
    return False


def log_has_hard_quota_error(log_path: Path) -> bool:
    """Read CLI error records and clear account-limit messages, not bare 429s."""
    try:
        with log_path.open() as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    # Some CLIs emit plain text instead of structured errors.
                    if is_hard_quota_error(line):
                        return True
                    continue
                if not isinstance(event, dict):
                    continue
                if (event.get("api_error_status") == 429
                        and is_hard_quota_error(event.get("api_error"))):
                    return True
                if event.get("error") and is_hard_quota_error(event["error"]):
                    return True
                if event.get("type") in ("error", "turn.failed") and is_hard_quota_error(event):
                    return True
    except OSError:
        pass
    return False
