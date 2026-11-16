"""An offline tour of toolrouter: no model, no network. Run it:

    uv run python examples/quickstart.py

A real model would produce the `message` below; everything after that is what toolrouter does.
"""

import asyncio

from tools import convert_currency, get_forecast, read_note, send_email

from toolrouter import LoggingHook, ReadOnlyPolicy, ToolRegistry, ToolRouter
from toolrouter.feedback import model_text
from toolrouter.formats import ollama

registry = ToolRegistry([get_forecast, convert_currency, read_note, send_email])


def model_call(name: str, **arguments: object) -> dict[str, object]:
    return {"function": {"name": name, "arguments": arguments}}


# Pretend a model answered with these tool calls (the Ollama chat message shape).
message = {
    "role": "assistant",
    "tool_calls": [
        model_call("get_forecast", city="Oslo", units="f"),  # fine
        model_call("get_forecast", city="Atlantis"),  # the tool itself says no
        model_call("convert_currency", amount="100", source="USD", target="NOK"),  # "100" is a str
        model_call("read_note", path="../../etc/passwd"),  # tries to leave the notes folder
        model_call("send_email", to="a@b.co", subject="hi", body="x", api_key="sk-123"),  # WRITE
        model_call("get_forecats"),  # a typo
    ],
}


async def main() -> None:
    router = ToolRouter(
        registry,
        policy=ReadOnlyPolicy(),  # this session may read, never write
        hooks=[LoggingHook()],  # one log line per call; secrets are redacted
    )
    calls = ollama.parse_calls(message)
    results = await router.dispatch_many(calls)  # concurrent, results in call order

    for call, result in zip(calls, results, strict=True):
        print(f"{call.name}\n  -> {model_text(result)}\n")

    # What you would append to the conversation before asking the model again:
    print(f"{len(ollama.result_messages(results))} tool messages ready to send back")


if __name__ == "__main__":
    asyncio.run(main())
