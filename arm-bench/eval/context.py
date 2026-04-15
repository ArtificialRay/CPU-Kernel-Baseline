import copy
import json

# ─── History compression ────────────────────────────────────────────────────

def _compress_history_with_code(
    messages: list[dict],
    keep_full_turns: int = 2,
    latest_perf_code: str | None = None,
) -> list[dict]:
    """
    Compress old turns to keep context size bounded.

    The last `keep_full_turns` complete assistant+tool pairs are kept verbatim.
    Older turns have large payloads replaced with compact summaries.

    If `latest_perf_code` is provided, the ```cpp block in the initial user
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

    # Replace the ```cpp block in the initial user message with the latest
    # perf'd code so the LLM treats it as the current implementation.
    if latest_perf_code is not None:
        for i, msg in enumerate(result):
            if i == 1 and msg["role"] == "user":
                content = msg["content"]
                start_marker = "```cpp\n"
                end_marker = "\n```"
                start_idx = content.find(start_marker)
                if start_idx != -1:
                    end_idx = content.find(end_marker, start_idx + len(start_marker))
                    if end_idx != -1:
                        msg = copy.deepcopy(msg)
                        msg["content"] = (
                            content[: start_idx + len(start_marker)]
                            + latest_perf_code
                            + content[end_idx:]
                        )
                        result[i] = msg
                break

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