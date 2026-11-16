"""Module-level sample tools (type hints must be resolvable, so not defined inside tests)."""

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import Field

from toolrouter import SideEffect, StrictModel, tool


class Color(StrEnum):
    """A color."""

    RED = "red"
    BLUE = "blue"


class Address(StrictModel):
    street: str
    city: str = Field(description="City name.")
    zip: str | None = None


@tool(side_effect=SideEffect.READ)
def get_weather(city: str, units: Literal["c", "f"] = "c") -> str:
    """Current weather for a city.

    Args:
        city: Name of the city.
        units: Temperature units.
    """
    return f"{city}:{units}"


@tool
def search(
    query: Annotated[str, Field(min_length=1, max_length=200)],
    limit: Annotated[int, Field(ge=1, le=50)] = 10,
    tags: list[str] | None = None,
) -> list[str]:
    """Search documents."""
    return []


@tool(side_effect=SideEffect.WRITE)
def ship(address: Address, extras: list[Address], color: Color = Color.RED) -> str:
    """Ship a parcel."""
    return address.city


@tool
def now() -> str:
    """Current time."""
    return "t"


@tool
def title_tool(title: str, properties: int = 0) -> str:
    """Has fields named like schema keywords."""
    return title


ALL = [get_weather, search, ship, now, title_tool]


@tool
def blob(meta: dict[str, Any]) -> str:
    """Takes free-form metadata."""
    return ""
