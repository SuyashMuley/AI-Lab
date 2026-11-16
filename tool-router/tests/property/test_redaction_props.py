"""Plant a secret anywhere a secret field can be: it must not appear in ANY observable output."""

import asyncio
import json
import logging
from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from tests.fakes import secret_tools as tools
from tests.fakes.secret_tools import SENTINEL
from tests.fuzz import fuzz
from toolrouter import (
    LoggingHook,
    RecordingHook,
    ToolCall,
    ToolRegistry,
    ToolRouter,
)
from toolrouter.feedback import model_text
from toolrouter.formats import anthropic, ollama, openai

REGISTRY = ToolRegistry(tools.ALL)

harmless = st.text(max_size=12).filter(lambda s: "ZXQV" not in s)
noise = st.recursive(
    st.none() | st.booleans() | st.integers() | harmless,
    lambda c: st.lists(c, max_size=3) | st.dictionaries(harmless, c, max_size=3),
    max_leaves=8,
)
# Where the secret might be planted: right type, wrong type, wrapped in noise.
secret_values = st.one_of(
    st.just(SENTINEL),
    st.just([SENTINEL]),
    st.just({"k": SENTINEL}),
    st.just(SENTINEL + "x" * 300),
    st.just(12345),  # wrong type: a validation issue on a secret field
)

plants = st.one_of(
    st.builds(lambda s, n: {"user": n, "password": s}, secret_values, harmless),
    st.builds(lambda s, n: {"url": n, "api_key": s}, secret_values, harmless),
    st.builds(lambda s, n: {"note": n, "key": s}, secret_values, harmless),
    st.builds(lambda s, n: {"label": n, "headers": {"h": s}}, secret_values, harmless),
    st.builds(
        lambda s, n: {"creds": {"user": n, "password": s}, "items": []}, secret_values, harmless
    ),
    st.builds(
        lambda s, n: {"creds": {"user": n, "password": "ok"}, "items": [{"name": n, "token": s}]},
        secret_values,
        harmless,
    ),
    st.builds(lambda s: {"tokens": [s, s]}, secret_values),
    st.builds(lambda s, n: {"target": {"name": n, "token": s}}, secret_values, harmless),
    st.builds(lambda s, n: {"note": n, "token": s}, secret_values, harmless),
)
names = st.sampled_from([t.name for t in tools.ALL] + ["no_such_tool"])


@fuzz(400)
@given(names, plants, noise, st.booleans())
def test_a_planted_secret_never_leaks(
    name: str, args: dict[str, Any], extra: Any, as_text: bool
) -> None:
    secret_text = SENTINEL
    arguments: Any = json.dumps(args) if as_text else args
    call = ToolCall(id="c", name=name, arguments=arguments)

    recorder = RecordingHook()
    log_lines: list[str] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            log_lines.append(record.getMessage())

    logger = logging.getLogger("test.redaction.props")
    logger.setLevel(logging.DEBUG)
    handler = Capture()
    logger.addHandler(handler)
    try:
        router = ToolRouter(REGISTRY, hooks=[recorder, LoggingHook(logger)])
        result = asyncio.run(router.dispatch(call))
    finally:
        logger.removeHandler(handler)

    observable = [
        json.dumps(recorder.before_events[0].arguments, default=repr),
        json.dumps(recorder.after_events[0].call.arguments, default=repr),
        result.model_dump_json(),
        model_text(result),
        json.dumps(anthropic.result_messages([result])),
        json.dumps(openai.result_messages([result])),
        json.dumps(ollama.result_messages([result])),
        *log_lines,
    ]
    for text in observable:
        assert secret_text not in text, text
    # ...and nothing the model is shown about the *tools* contains a secret default either.
    for t in tools.ALL:
        assert "hunter2-default" not in json.dumps(anthropic.tool(t))
