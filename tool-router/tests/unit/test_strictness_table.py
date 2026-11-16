"""The documented behavior of strict vs coerce validation, one row per (type, JSON input).

Strict is Pydantic JSON-mode strict: JSON strings are fine for enums/datetimes/UUIDs/paths,
but no guessing across JSON types ("42" is not an int, 1 is not a bool).
Coerce is lax mode: it converts across JSON types where Pydantic can.
"""

import datetime
import decimal
import uuid
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Optional

import pytest

from tests.fakes.factories import make_tool
from toolrouter import StrictModel, ToolCallError, validate_call
from toolrouter.errors import ErrorCode


class Color(StrEnum):
    RED = "red"
    BLUE = "blue"


class Inner(StrictModel):
    a: int


REJECT = object()
UID = uuid.UUID("12345678-1234-5678-1234-567812345678")

# (annotation, raw JSON for x, strict result, coerce result)
TABLE: list[tuple[Any, str, Any, Any]] = [
    # ints
    (int, "42", 42, 42),
    (int, '"42"', REJECT, 42),
    (int, "1.0", REJECT, 1),
    (int, "1.5", REJECT, REJECT),
    (int, "1e2", REJECT, 100),
    (int, "true", REJECT, 1),
    # floats
    (float, "3", 3.0, 3.0),
    (float, "3.5", 3.5, 3.5),
    (float, '"3.5"', REJECT, 3.5),
    # bools
    (bool, "true", True, True),
    (bool, '"true"', REJECT, True),
    (bool, "1", REJECT, True),
    (bool, '"yes"', REJECT, True),
    # strings never coerce from other JSON types
    (str, '"x"', "x", "x"),
    (str, "5", REJECT, REJECT),
    (str, "null", REJECT, REJECT),
    # enums and literals
    (Color, '"red"', Color.RED, Color.RED),
    (Color, '"green"', REJECT, REJECT),
    (Literal["a", "b"], '"a"', "a", "a"),
    (Literal[1, 2], "1", 1, 1),
    (Literal[1, 2], '"1"', REJECT, REJECT),
    # containers
    (list[int], "[1, 2]", [1, 2], [1, 2]),
    (list[int], '["1"]', REJECT, [1]),
    (list[int], "1", REJECT, REJECT),
    (tuple[int, int], "[1, 2]", (1, 2), (1, 2)),
    (dict[str, int], '{"k": 1}', {"k": 1}, {"k": 1}),
    (dict[str, int], '{"k": "1"}', REJECT, {"k": 1}),
    # nested models
    (Inner, '{"a": 1}', Inner(a=1), Inner(a=1)),
    (Inner, '{"a": "1"}', REJECT, Inner(a=1)),
    (Inner, '{"a": 1, "extra": 2}', REJECT, REJECT),  # unknown fields never allowed
    # strings that legitimately encode richer types are fine in strict JSON mode
    (
        datetime.datetime,
        '"2024-01-02T03:04:05"',
        datetime.datetime(2024, 1, 2, 3, 4, 5),
        datetime.datetime(2024, 1, 2, 3, 4, 5),
    ),
    (datetime.date, '"2024-01-02"', datetime.date(2024, 1, 2), datetime.date(2024, 1, 2)),
    (uuid.UUID, '"12345678-1234-5678-1234-567812345678"', UID, UID),
    (decimal.Decimal, '"1.5"', decimal.Decimal("1.5"), decimal.Decimal("1.5")),
    (Path, '"a/b"', Path("a/b"), Path("a/b")),
    # optionals
    (Optional[int], "null", None, None),  # noqa: UP045
    (Optional[int], '"3"', REJECT, 3),  # noqa: UP045
    (int, "null", REJECT, REJECT),
]


def ids() -> list[str]:
    return [f"{getattr(a, '__name__', str(a))}<-{raw}" for a, raw, _, _ in TABLE]


def run(annotation: Any, raw: str, *, router_coerce: bool) -> Any:
    t = make_tool(annotation)
    try:
        return validate_call(t, f'{{"x": {raw}}}', router_coerce=router_coerce).x  # type: ignore[attr-defined]
    except ToolCallError as exc:
        assert exc.info.code is ErrorCode.ARGUMENTS_INVALID
        return REJECT


@pytest.mark.parametrize(("annotation", "raw", "strict", "_"), TABLE, ids=ids())
def test_strict(annotation: Any, raw: str, strict: Any, _: Any) -> None:
    assert run(annotation, raw, router_coerce=False) == strict


@pytest.mark.parametrize(("annotation", "raw", "_", "coerce"), TABLE, ids=ids())
def test_coerce(annotation: Any, raw: str, _: Any, coerce: Any) -> None:
    assert run(annotation, raw, router_coerce=True) == coerce


def test_missing_fields_are_rejected_in_both_modes() -> None:
    t = make_tool(int)
    for coerce in (False, True):
        with pytest.raises(ToolCallError) as ei:
            validate_call(t, "{}", router_coerce=coerce)
        assert ei.value.info.issues[0].type == "missing"


def test_dict_input_behaves_like_json_text() -> None:
    t = make_tool(Color)
    assert validate_call(t, {"x": "red"}).x is Color.RED  # type: ignore[attr-defined]
    t2 = make_tool(int)
    with pytest.raises(ToolCallError):
        validate_call(t2, {"x": "42"})
    assert validate_call(t2, {"x": "42"}, router_coerce=True).x == 42  # type: ignore[attr-defined]
