import dataclasses
from typing import Annotated, Any, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from toolrouter import SideEffect, StrictModel, Tool, ToolDefinitionError, tool

# ---- signature-derived tools ---------------------------------------------------------------


@tool
def get_weather(city: str, units: Literal["c", "f"] = "c") -> str:
    """Current weather for a city.

    Args:
        city: Name of the city.
        units: Temperature units.
    """
    return f"{city}:{units}"


def test_bare_decorator_builds_tool() -> None:
    assert isinstance(get_weather, Tool)
    assert get_weather.name == "get_weather"
    assert get_weather.description == "Current weather for a city."
    assert get_weather.side_effect is SideEffect.NONE
    assert get_weather.timeout_s == 30.0
    assert get_weather.coerce is None
    assert get_weather.result_type is str
    assert not get_weather.is_async


def test_args_model_fields_defaults_and_descriptions() -> None:
    fields = get_weather.args_model.model_fields
    assert list(fields) == ["city", "units"]
    assert fields["city"].is_required()
    assert fields["units"].default == "c"
    assert fields["city"].description == "Name of the city."
    assert fields["units"].description == "Temperature units."


def test_args_model_rejects_unknown_fields_and_is_frozen() -> None:
    model = get_weather.args_model
    ok = model.model_validate({"city": "Oslo"})
    assert ok.units == "c"  # type: ignore[attr-defined]
    with pytest.raises(ValidationError, match="extra_forbidden"):
        model.model_validate({"city": "Oslo", "bogus": 1})
    with pytest.raises(ValidationError):
        ok.city = "x"  # type: ignore[attr-defined]


def test_options_override_name_description_and_limits() -> None:
    @tool(
        name="weather",
        description="Explicit description.",
        side_effect=SideEffect.READ,
        timeout_s=5,
        max_output_chars=100,
        coerce=True,
    )
    def f(city: str) -> str:
        """Docstring that loses."""
        return city

    assert (f.name, f.description) == ("weather", "Explicit description.")
    assert f.side_effect is SideEffect.READ
    assert (f.timeout_s, f.max_output_chars, f.coerce) == (5, 100, True)


def test_annotated_constraints_and_own_descriptions_are_kept() -> None:
    @tool
    def search(
        q: Annotated[str, Field(min_length=1, description="From the annotation.")],
        limit: Annotated[int, Field(ge=1, le=50)] = 10,
    ) -> list[str]:
        """Search.

        Args:
            q: From the docstring (should lose).
            limit: Max results.
        """
        return []

    fields = search.args_model.model_fields
    assert fields["q"].description == "From the annotation."
    assert fields["limit"].description == "Max results."
    with pytest.raises(ValidationError):
        search.args_model.model_validate({"q": ""})
    with pytest.raises(ValidationError):
        search.args_model.model_validate({"q": "x", "limit": 0})


def test_nested_models_are_preserved_and_must_forbid_extras() -> None:
    class Address(StrictModel):
        city: str

    @tool
    def ship(address: Address) -> str:
        """Ship it."""
        return address.city

    with pytest.raises(ValidationError):
        ship.args_model.model_validate({"address": {"city": "x", "zip": "1"}})


async def test_invoke_passes_nested_model_instances_not_dicts() -> None:
    class Address(StrictModel):
        city: str

    seen: list[Any] = []

    @tool
    def ship(address: Address) -> str:
        """Ship it."""
        seen.append(address)
        return address.city

    args = ship.args_model.model_validate({"address": {"city": "Oslo"}})
    assert await ship.invoke(args) == "Oslo"
    assert isinstance(seen[0], Address)


def test_nested_model_that_allows_extras_is_rejected() -> None:
    class Loose(BaseModel):  # default extra="ignore"
        city: str

    with pytest.raises(ToolDefinitionError, match="accept unknown fields"):

        @tool
        def ship(address: Loose) -> str:
            """Ship it."""
            return ""


def test_free_form_dict_is_allowed() -> None:
    @tool
    def tag(meta: dict[str, Any]) -> str:
        """Tag it."""
        return ""

    assert tag.args_model.model_validate({"meta": {"a": 1}})


async def test_async_handler_and_invoke() -> None:
    @tool
    async def add(a: int, b: int = 1) -> int:
        """Add numbers."""
        return a + b

    assert add.is_async
    assert await add.invoke(add.args_model.model_validate({"a": 2})) == 3


async def test_sync_invoke_and_direct_call() -> None:
    assert await get_weather.invoke(get_weather.args_model.model_validate({"city": "X"})) == "X:c"
    assert get_weather("Y", "f") == "Y:f"  # __call__ goes straight to the function


async def test_invoke_rejects_wrong_model_type() -> None:
    class NotArgs(BaseModel):
        city: str

    with pytest.raises(TypeError, match="expected"):
        await get_weather.invoke(NotArgs(city="x"))


def test_return_annotation_variants() -> None:
    @tool
    def a() -> None:
        """No result."""

    @tool
    def b():  # type: ignore[no-untyped-def]
        """Unannotated."""

    assert a.result_type is type(None)
    assert b.result_type is None


def test_no_parameters_is_fine() -> None:
    @tool
    def now() -> str:
        """Current time."""
        return "t"

    assert now.args_model.model_validate({}) is not None
    with pytest.raises(ValidationError):
        now.args_model.model_validate({"x": 1})


# ---- explicit-model tools ------------------------------------------------------------------


class SendArgs(StrictModel):
    to: str
    body: str = ""


def test_explicit_model_tool() -> None:
    @tool(model=SendArgs, name="send", description="Send a message.")
    def send(args: SendArgs) -> str:
        return args.to

    assert send.args_model is SendArgs
    assert send.takes_model


async def test_explicit_model_handler_receives_the_model() -> None:
    @tool(model=SendArgs)
    async def send_message(args: SendArgs) -> str:
        """Send a message."""
        return f"{args.to}|{args.body}"

    assert await send_message.invoke(SendArgs(to="a", body="b")) == "a|b"


def test_explicit_model_must_forbid_extras() -> None:
    class Loose(BaseModel):
        x: int

    with pytest.raises(ToolDefinitionError, match="accept unknown fields"):

        @tool(model=Loose, name="t", description="d")
        def t(args: Loose) -> None: ...

    class Open(BaseModel):
        model_config = ConfigDict(extra="allow")
        x: int

    with pytest.raises(ToolDefinitionError, match="accept unknown fields"):
        tool(model=Open, name="t", description="d")(lambda args: None)


def test_explicit_model_handler_signature_checks() -> None:
    with pytest.raises(ToolDefinitionError, match="exactly one parameter"):

        @tool(model=SendArgs, name="t", description="d")
        def two(a: SendArgs, b: int) -> None: ...

    with pytest.raises(ToolDefinitionError, match="exactly one parameter"):

        @tool(model=SendArgs, name="t", description="d")
        def zero() -> None: ...

    with pytest.raises(ToolDefinitionError, match="annotated"):

        @tool(model=SendArgs, name="t", description="d")
        def wrong(args: int) -> None: ...

    with pytest.raises(ToolDefinitionError, match="Pydantic model class"):
        tool(model=dict, name="t", description="d")(lambda a: a)  # type: ignore[arg-type]


# ---- definition-time rejections ------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "1abc", "has space", "dot.name", "a" * 65, "émoji", "x/y"])
def test_invalid_names_rejected(bad: str) -> None:
    with pytest.raises(ToolDefinitionError, match="Invalid tool name"):
        tool(name=bad, description="d")(lambda: None)


@pytest.mark.parametrize("good", ["a", "get_weather", "get-weather", "A1_b-2", "a" * 64])
def test_valid_names_accepted(good: str) -> None:
    assert tool(name=good, description="d")(lambda: None).name == good


def test_lambda_needs_explicit_name() -> None:
    with pytest.raises(ToolDefinitionError, match="explicit name"):
        tool(description="d")(lambda: None)


def test_missing_description_rejected() -> None:
    def nodoc(x: int) -> None: ...

    with pytest.raises(ToolDefinitionError, match="needs a description"):
        tool(nodoc)
    with pytest.raises(ToolDefinitionError, match="needs a description"):
        tool(description="   ")(nodoc)


def test_docstring_with_only_args_has_no_description() -> None:
    def f(x: int) -> None:
        """
        Args:
            x: the x.
        """

    with pytest.raises(ToolDefinitionError, match="needs a description"):
        tool(f)


def test_unannotated_parameter_rejected() -> None:
    def f(x) -> None:  # type: ignore[no-untyped-def]
        """Doc."""

    with pytest.raises(ToolDefinitionError, match="missing type annotation"):
        tool(f)


def test_varargs_kwargs_and_positional_only_rejected() -> None:
    def a(*args: int) -> None:
        """Doc."""

    def b(**kwargs: int) -> None:
        """Doc."""

    def c(x: int, /) -> None:
        """Doc."""

    for fn in (a, b, c):
        with pytest.raises(ToolDefinitionError):
            tool(fn)


@pytest.mark.parametrize("pname", ["schema", "json", "copy", "model_dump", "_private"])
def test_reserved_parameter_names_rejected(pname: str) -> None:
    ns: dict[str, Any] = {}
    exec(f"def f({pname}: int) -> None:\n    'Doc.'\n", ns)
    with pytest.raises(ToolDefinitionError, match="reserved"):
        tool(ns["f"])


def test_unresolvable_annotation_rejected() -> None:
    ns: dict[str, Any] = {}
    exec("def f(x: 'DoesNotExist') -> None:\n    'Doc.'\n", ns)
    with pytest.raises(ToolDefinitionError, match="cannot resolve type hints"):
        tool(ns["f"])


def test_non_schema_able_type_rejected() -> None:
    class Opaque:
        pass

    with pytest.raises(ToolDefinitionError):

        @tool
        def f(x: Opaque) -> None:
            """Doc."""


def test_non_callable_rejected() -> None:
    with pytest.raises(ToolDefinitionError, match="needs a function"):
        tool(42)  # type: ignore[call-overload]


@pytest.mark.parametrize("bad", [0, -1, float("inf"), float("nan"), True])
def test_bad_timeouts_rejected(bad: float) -> None:
    with pytest.raises(ToolDefinitionError, match="timeout_s"):
        tool(name="t", description="d", timeout_s=bad)(lambda: None)


@pytest.mark.parametrize("bad", [0, -5, 1.5, True])
def test_bad_max_output_rejected(bad: Any) -> None:
    with pytest.raises(ToolDefinitionError, match="max_output_chars"):
        tool(name="t", description="d", max_output_chars=bad)(lambda: None)


def test_bad_side_effect_rejected() -> None:
    with pytest.raises(ValueError):
        tool(name="t", description="d", side_effect="launch-missiles")(lambda: None)  # type: ignore[call-overload]


def test_tool_is_immutable() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        get_weather.name = "other"  # type: ignore[misc]
