import pytest

from tests.fakes.sample_tools import ALL, get_weather, now, search
from toolrouter import (
    DuplicateToolError,
    StrictModel,
    Tool,
    ToolDefinitionError,
    ToolRegistry,
    tool,
)


def test_register_get_and_order() -> None:
    reg = ToolRegistry()
    assert len(reg) == 0
    assert reg.register(get_weather) is get_weather
    reg.register(search)
    assert reg.get("get_weather") is get_weather
    assert reg["search"] is search
    assert reg.get("nope") is None
    assert "search" in reg
    assert "nope" not in reg
    assert 42 not in reg
    assert reg.names() == ["get_weather", "search"]
    assert list(reg) == [get_weather, search]
    assert len(reg) == 2


def test_constructor_registers_all() -> None:
    reg = ToolRegistry(ALL)
    assert reg.names() == [t.name for t in ALL]


def test_duplicate_names_rejected_and_registry_unchanged() -> None:
    reg = ToolRegistry([get_weather])
    other = tool(name="get_weather", description="Another one.")(lambda: None)
    with pytest.raises(DuplicateToolError, match="get_weather") as ei:
        reg.register(other)
    assert ei.value.name == "get_weather"
    assert isinstance(ei.value, ToolDefinitionError)
    assert reg["get_weather"] is get_weather
    assert len(reg) == 1


def test_duplicate_in_constructor_rejected() -> None:
    with pytest.raises(DuplicateToolError):
        ToolRegistry([now, now])


def test_non_tool_rejected_with_helpful_message() -> None:
    def forgot_decorator() -> None:
        """Doc."""

    with pytest.raises(ToolDefinitionError, match="forget the @tool"):
        ToolRegistry().register(forgot_decorator)  # type: ignore[arg-type]


def test_unregister() -> None:
    reg = ToolRegistry([now, search])
    assert reg.unregister("now") is now
    assert reg.names() == ["search"]
    with pytest.raises(KeyError, match="No tool named"):
        reg.unregister("now")
    reg.register(now)  # name is free again


def test_missing_lookup_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="No tool named 'x'"):
        ToolRegistry()["x"]
    with pytest.raises(KeyError):
        ToolRegistry().schema("x")


def test_iteration_is_a_snapshot_safe_to_mutate_during() -> None:
    reg = ToolRegistry([now, search])
    for t in reg:
        reg.unregister(t.name)
    assert len(reg) == 0


def test_schema_returns_independent_copies() -> None:
    reg = ToolRegistry([get_weather])
    first = reg.schema("get_weather")
    first["properties"]["city"]["type"] = "MUTATED"
    first["properties"]["injected"] = {}
    assert reg.schema("get_weather")["properties"]["city"]["type"] == "string"
    assert "injected" not in reg.schema("get_weather")["properties"]


class Node(StrictModel):
    children: list["Node"]


def test_recursive_args_rejected_at_registration() -> None:
    @tool
    def walk(root: Node) -> str:
        """Walk a tree."""
        return ""

    assert isinstance(walk, Tool)  # defining it is fine...
    with pytest.raises(ToolDefinitionError, match="recursive model 'Node'"):
        ToolRegistry().register(walk)  # ...registering fails fast
