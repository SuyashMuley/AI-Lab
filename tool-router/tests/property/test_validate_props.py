"""Fuzz the validator: arbitrary model output must never crash it or reach a handler unvalidated."""

import json
from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from tests.fakes.sample_tools import ALL, blob
from tests.fuzz import fuzz
from toolrouter import ToolCallError, validate_call
from toolrouter.errors import ErrorCode

CLIENT_ERRORS = {
    ErrorCode.INVALID_JSON,
    ErrorCode.ARGUMENTS_TOO_LARGE,
    ErrorCode.ARGUMENTS_INVALID,
}

json_values = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=True, allow_infinity=True)
    | st.text(),
    lambda children: (
        st.lists(children, max_size=4) | st.dictionaries(st.text(max_size=8), children, max_size=4)
    ),
    max_leaves=30,
)
tools = st.sampled_from([*ALL, blob])


def outcome(tool: Any, arguments: Any, **kw: Any) -> None:
    """Either a valid instance of the args model, or a *client-caused* ToolCallError."""
    try:
        result = validate_call(tool, arguments, **kw)
    except ToolCallError as exc:
        assert exc.info.code in CLIENT_ERRORS, exc.info
        assert exc.info.message
        assert exc.info.retryable
    else:
        assert isinstance(result, tool.args_model)


@fuzz(400)
@given(tools, json_values, st.booleans())
def test_any_json_value_as_text(tool: Any, value: Any, coerce: bool) -> None:
    try:
        text = json.dumps(value)
    except (ValueError, RecursionError):
        return
    outcome(tool, text, router_coerce=coerce)


@fuzz(400)
@given(tools, st.dictionaries(st.text(max_size=8), json_values, max_size=5), st.booleans())
def test_any_dict(tool: Any, arguments: dict[str, Any], coerce: bool) -> None:
    outcome(tool, arguments, router_coerce=coerce)


@fuzz(400)
@given(tools, st.text(alphabet=st.characters(codec="utf-8", exclude_categories=()), max_size=200))
def test_arbitrary_text(tool: Any, text: str) -> None:
    outcome(tool, text)


@fuzz(300)
@given(tools, st.binary(max_size=200))
def test_arbitrary_bytes_as_lossy_text(tool: Any, data: bytes) -> None:
    outcome(tool, data.decode("utf-8", errors="surrogateescape"))
    outcome(tool, data)  # raw bytes are rejected, not crashed on


@fuzz(200)
@given(tools, st.text(max_size=100), st.text(max_size=100))
def test_json_text_with_mangled_tail(tool: Any, head: str, tail: str) -> None:
    outcome(tool, json.dumps({"city": head}) + tail)
    outcome(tool, tail + json.dumps({"city": head}))


@fuzz(200)
@given(st.integers(min_value=1, max_value=400))
def test_nesting_depth_is_always_bounded(depth: int) -> None:
    text = '{"meta": ' + "[" * depth + "]" * depth + "}"
    outcome(blob, text)
    # beyond the default depth limit it must be rejected, never accepted
    if depth + 1 > 32:
        try:
            validate_call(blob, text)
        except ToolCallError as exc:
            assert exc.info.code is ErrorCode.ARGUMENTS_TOO_LARGE
        else:
            raise AssertionError("over-deep payload was accepted")
