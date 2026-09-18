"""eval/llm_call.py — wire-API dispatch for Path 1's litellm agent loop.

litellm.completion() (Chat Completions) is the default. Models listed in
config/kernel_contracts.yaml's models_using_responses_api (currently
gpt-5.6-luna) reject tool-calling + reasoning there and need
litellm.responses() (Responses API) instead — a different request/response
shape (flat `input`/`output` items vs `messages`/`choices[0].message`).

Translation happens only at this call boundary — chat-shaped
completion_kwargs in, a message duck-typed like litellm's
ChatCompletionMessage out — so evaluator.py's turn loop (history
compression, notepad, tool-call sanitization, version tracking) only ever
deals with one shape, regardless of which wire API served a given turn.

Known limitation: So far, reasoning content from a Responses API turn isn't
round-tripped into the next turn's request 
"""

from types import SimpleNamespace

import litellm

from contracts import AGENT_LOOP_DEFAULTS


def wire_api_for(model: str) -> str:
    """"chat" (default, litellm.completion()) or "responses" (litellm.responses(),
    for models that reject tool-calling + reasoning on Chat Completions)."""
    if any(m in model for m in AGENT_LOOP_DEFAULTS["models_using_responses_api"]):
        return "responses"
    return "chat"


def _to_responses_kwargs(completion_kwargs: dict) -> dict:
    """Chat-shaped completion_kwargs (messages/tools) -> litellm.responses() kwargs
    (input/tools). Tool calls and their results are flattened into standalone
    function_call / function_call_output items — the Responses API doesn't nest
    them inside a message the way Chat Completions does."""
    input_items = []
    for msg in completion_kwargs["messages"]:
        if msg["role"] == "tool":
            input_items.append({
                "type": "function_call_output",
                "call_id": msg["tool_call_id"],
                "output": msg["content"],
            })
        elif msg["role"] == "assistant" and msg.get("tool_calls"):
            if msg.get("content"):
                input_items.append({"role": "assistant", "content": msg["content"]})
            for tc in msg["tool_calls"]:
                input_items.append({
                    "type": "function_call",
                    "call_id": tc["id"],
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                })
        else:
            input_items.append({"role": msg["role"], "content": msg["content"]})

    tools = [
        {
            "type": "function",
            "name": t["function"]["name"],
            "description": t["function"].get("description", ""),
            "parameters": t["function"]["parameters"],
        }
        for t in completion_kwargs.get("tools", [])
    ]

    responses_kwargs = {
        "model": completion_kwargs["model"],
        "input": input_items,
        "tools": tools,
        "tool_choice": completion_kwargs.get("tool_choice", "auto"),
    }
    for key in ("timeout", "temperature", "api_key", "api_base"):
        if key in completion_kwargs:
            responses_kwargs[key] = completion_kwargs[key]
    return responses_kwargs


def _normalize_responses_output(output: list) -> SimpleNamespace:
    """Responses API `response.output` items -> a message duck-typed like
    litellm's ChatCompletionMessage, so the turn loop needs no wire-API
    awareness downstream of the call."""
    items = [it.model_dump() if hasattr(it, "model_dump") else it for it in output]

    content = "".join(
        part.get("text", "")
        for item in items if item.get("type") == "message"
        for part in item.get("content", []) if part.get("type") == "output_text"
    ) or None

    tool_calls = [
        {"id": item["call_id"], "type": "function",
         "function": {"name": item["name"], "arguments": item["arguments"]}}
        for item in items if item.get("type") == "function_call"
    ]

    dumped = {"role": "assistant", "content": content, "tool_calls": tool_calls or None}
    return SimpleNamespace(
        content=content,
        tool_calls=[
            SimpleNamespace(id=tc["id"], type="function",
                             function=SimpleNamespace(**tc["function"]))
            for tc in tool_calls
        ],
        model_dump=lambda: dumped,
    )


def call_llm(completion_kwargs: dict):
    """Runs one turn's LLM call, returning a message object duck-typed like
    litellm's ChatCompletionMessage (.content, .tool_calls, .model_dump())
    regardless of which wire API actually served it. Raises the same
    litellm.* exceptions either way, so callers' retry handling doesn't
    need to know which path ran."""
    if wire_api_for(completion_kwargs["model"]) == "chat":
        return litellm.completion(**completion_kwargs).choices[0].message
    response = litellm.responses(**_to_responses_kwargs(completion_kwargs))
    return _normalize_responses_output(response.output)
