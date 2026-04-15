import copy
import json

# ─── Last-round state injection ─────────────────────────────────────────────

_LAST_ROUND_MARKER = "\n<!-- last_round_state -->"


def _build_last_round_note(state: dict) -> str:
    """
    Build a short human-readable note about the previous tool call to inject
    into the initial user prompt so the LLM always sees the last round outcome.

    ``state`` keys:
      action        : "compile" | "run" | "perf"
      turn          : int
      success       : bool   (compile only)
      code          : str    (compile failure only — the code that failed)
      errors        : str    (compile failure only)
      correct       : bool   (run only)
      candidate_ms  : float  (run only)
      baseline_ms   : float  (run only)
      speedup       : float  (run / perf)
    """
    action = state.get("action")
    turn = state.get("turn", "?")

    if action == "compile":
        if not state.get("success"):
            errors = (state.get("errors") or "")[:400]
            code = state.get("code", "")
            return (
                f"{_LAST_ROUND_MARKER}\n"
                f"⚠ Last round (turn {turn}): compile FAILED. "
                f"The following code did NOT compile:\n"
                f"```cpp\n{code}\n```\n"
                f"Errors: {errors}\n"
                f"The current candidate above reflects the last successfully compiled version."
            )
        return f"{_LAST_ROUND_MARKER}\nLast round (turn {turn}): compile succeeded."

    elif action == "run":
        correct = state.get("correct")
        c_ms = state.get("candidate_ms")
        b_ms = state.get("baseline_ms")
        speedup = state.get("speedup")
        note = f"{_LAST_ROUND_MARKER}\nLast round (turn {turn}): run — correct={correct}"
        if c_ms is not None:
            note += f", candidate={c_ms} ms"
        if b_ms is not None:
            note += f", baseline={b_ms} ms"
        if speedup is not None:
            note += f", speedup={speedup}x"
        return note

    elif action == "perf":
        speedup = state.get("speedup")
        return f"{_LAST_ROUND_MARKER}\nLast round (turn {turn}): perf — speedup={speedup}x"

    return ""


# ─── History compression ────────────────────────────────────────────────────

def _compress_history_with_code(
    messages: list[dict],
    keep_full_turns: int = 2,
    latest_compiled_code: str | None = None,
    last_round_state: dict | None = None,
) -> list[dict]:
    """
    Compress old turns to keep context size bounded.

    The last `keep_full_turns` complete assistant+tool pairs are kept verbatim.
    Older turns have large payloads replaced with compact summaries.

    If `latest_compiled_code` is provided, the ```cpp block in the initial user
    message is replaced with that code so the LLM always sees the most recently
    perf'd implementation as the "current" candidate source.
    """
    assistant_indices = [i for i, m in enumerate(messages) if m["role"] == "assistant"]

    if len(assistant_indices) <= keep_full_turns:
        result = list(messages)
    else:
        # Build map: tool_call_id → compile success (True/False/None)
        compile_success: dict[str, bool] = {}
        for msg in messages:
            if msg["role"] == "tool":
                try:
                    content = json.loads(msg["content"])
                    if "success" in content:
                        compile_success[msg["tool_call_id"]] = content["success"]
                except (json.JSONDecodeError, KeyError):
                    pass

        keep_from = assistant_indices[-keep_full_turns]

        result = []
        for i, msg in enumerate(messages):
            if i < keep_from and i >= 2:
                msg = copy.deepcopy(msg)
                if msg["role"] == "assistant" and msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        if tc["function"]["name"] in ("compile", "submit"):
                            if compile_success.get(tc["id"], True): # only add successful compiled code
                                try:
                                    args = json.loads(tc["function"]["arguments"])
                                    code = args.get("code", "")
                                    if len(code) > 200: # reduce code length to less than 200
                                        args["code"] = (
                                            "/* [prior attempt: "
                                            f"{len(code)} chars omitted] */"
                                        )
                                        tc["function"]["arguments"] = json.dumps(args) # TODO: only assistant message has older code
                                except (json.JSONDecodeError, KeyError):
                                    pass
                elif msg["role"] == "tool":
                    try:
                        content = json.loads(msg["content"])
                        if "asm" in content and len(content["asm"]) > 200:  # if use disassemble, add asm to the msg content
                            lines = content["asm"].count("\n")
                            content["asm"] = f"[{lines} lines — omitted from history]"
                            msg["content"] = json.dumps(content)
                    except (json.JSONDecodeError, KeyError):
                        pass
            result.append(msg)

    # Modify the initial user message (index 1):
    #   (1) replace the ```cpp block with the latest compiled code
    #   (2) append / replace the last-round state note
    user_msg_idx = next(
        (i for i, m in enumerate(result) if i == 1 and m["role"] == "user"), None
    )
    if user_msg_idx is not None and (latest_compiled_code is not None or last_round_state is not None):
        content = result[user_msg_idx]["content"]

        # (1) Replace cpp block
        if latest_compiled_code is not None:
            start_marker = "```cpp\n"
            end_marker = "\n```"
            start_idx = content.find(start_marker)
            if start_idx != -1:
                end_idx = content.find(end_marker, start_idx + len(start_marker))
                if end_idx != -1:
                    content = (
                        content[: start_idx + len(start_marker)]
                        + latest_compiled_code
                        + content[end_idx:]
                    )

        # (2) Strip old state note, then append new one
        if last_round_state is not None:
            marker_idx = content.find(_LAST_ROUND_MARKER)
            if marker_idx != -1:
                content = content[:marker_idx]
            state_note = _build_last_round_note(last_round_state)
            if state_note:
                content += state_note

        msg = copy.deepcopy(result[user_msg_idx])
        msg["content"] = content
        result[user_msg_idx] = msg

    return result

def _compress_history(messages: list[dict], keep_full_turns: int = 2) -> list[dict]:
    """
    Compress old turns to keep context size bounded.

    The last `keep_full_turns` complete assistant+tool pairs are kept verbatim.
    Older turns have large payloads replaced with compact summaries.
    """
    assistant_indices = [i for i, m in enumerate(messages) if m["role"] == "assistant"]

    if len(assistant_indices) <= keep_full_turns:
        return messages

    # Build map: tool_call_id → compile success (True/False/None)
    compile_success: dict[str, bool] = {}
    for msg in messages:
        if msg["role"] == "tool":
            try:
                content = json.loads(msg["content"])
                if "success" in content:
                    compile_success[msg["tool_call_id"]] = content["success"]
            except (json.JSONDecodeError, KeyError):
                pass

    keep_from = assistant_indices[-keep_full_turns]

    result = []
    for i, msg in enumerate(messages):
        if i < keep_from and i >= 2:
            msg = copy.deepcopy(msg)
            if msg["role"] == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    if tc["function"]["name"] in ("compile", "submit"):
                        if compile_success.get(tc["id"], True): # only replace successful compiled code
                            try:
                                args = json.loads(tc["function"]["arguments"])
                                code = args.get("code", "")
                                if len(code) > 200: # reduce large code to short placehoulder
                                    args["code"] = (
                                        "/* [prior attempt: "
                                        f"{len(code)} chars omitted] */"
                                    )
                                    tc["function"]["arguments"] = json.dumps(args) # TODO: only assistant message has older code
                            except (json.JSONDecodeError, KeyError):
                                pass
            elif msg["role"] == "tool":
                try:
                    content = json.loads(msg["content"])
                    if "asm" in content and len(content["asm"]) > 200:  # if use disassemble, add asm to the msg content
                        lines = content["asm"].count("\n")
                        content["asm"] = f"[{lines} lines — omitted from history]"
                        msg["content"] = json.dumps(content)
                except (json.JSONDecodeError, KeyError):
                    pass
        result.append(msg)
    return result