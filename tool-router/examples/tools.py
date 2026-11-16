"""Example tools to copy and adapt. Try them without any model:

    toolrouter list --tools examples/tools.py
    toolrouter call get_forecast --tools examples/tools.py --args '{"city": "Oslo"}'
    toolrouter chat "Is it warmer in Oslo or Cairo?" --tools examples/tools.py

Each tool shows one feature of the library; the comments say which.
"""

import asyncio
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from toolrouter import Confined, Secret, SideEffect, StrictModel, ToolError, tool

# ---- 1. A plain function. Every parameter needs a type annotation; the docstring becomes the
#         tool description and the "Args:" section becomes per-parameter descriptions. Returning a
#         Pydantic model is fine: it is validated against the annotation, then sent as JSON. -------


class Forecast(StrictModel):
    city: str
    temperature: float
    units: Literal["c", "f"]
    conditions: str


_FORECASTS = {
    "oslo": (4.0, "light snow"),
    "cairo": (28.0, "sunny"),
    "lima": (19.0, "overcast"),
}


@tool(side_effect=SideEffect.READ)  # READ: a read-only policy still allows this tool
def get_forecast(city: str, units: Literal["c", "f"] = "c") -> Forecast:
    """Today's forecast for a city.

    Args:
        city: City name, e.g. "Oslo".
        units: "c" for Celsius or "f" for Fahrenheit.
    """
    try:
        celsius, conditions = _FORECASTS[city.strip().lower()]
    except KeyError:
        known = ", ".join(sorted(c.title() for c in _FORECASTS))
        # ToolError: an *expected* failure. Its message is shown to the model, so make it useful.
        raise ToolError(f"No forecast for {city!r}. Known cities: {known}.") from None
    temperature = celsius if units == "c" else celsius * 9 / 5 + 32
    return Forecast(city=city.title(), temperature=temperature, units=units, conditions=conditions)


# ---- 2. Constraints with Annotated[..., Field(...)], and an async handler (awaited as-is; sync
#         handlers run in worker threads so they never block the event loop). ----

_RATES = {"USD": 1.0, "EUR": 0.92, "NOK": 10.8}


@tool
async def convert_currency(
    amount: Annotated[float, Field(gt=0, description="Amount to convert.")],
    source: Literal["USD", "EUR", "NOK"],
    target: Literal["USD", "EUR", "NOK"],
) -> str:
    """Convert an amount between currencies using fixed demo rates."""
    await asyncio.sleep(0)  # stand-in for a network call
    converted = amount / _RATES[source] * _RATES[target]
    return f"{amount:g} {source} = {converted:.2f} {target}"


# ---- 3. Confined: a path argument that cannot leave a directory (no "..", no absolute paths
#         outside it, no symlink escapes). The handler receives the resolved real path. ----

NOTES_DIR = Path(__file__).parent / "notes"
NotePath = Annotated[Path, Confined(NOTES_DIR)]


@tool(side_effect=SideEffect.READ)
def read_note(path: NotePath) -> str:
    """Read a note from the notes folder, e.g. "welcome.txt"."""
    if not path.is_file():
        raise ToolError(f"No such note: {path.name}")
    return path.read_text()


# ---- 4. An explicit Pydantic model (the handler receives it whole), a WRITE side effect (denied by
#         a read-only policy), and a secret (redacted from logs, hook events and error messages,
#         and never sent to the model as a default). ----


class EmailArgs(StrictModel):
    to: Annotated[str, Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$", description="Recipient.")]
    subject: str
    body: str
    api_key: Annotated[str, Secret()]


@tool(model=EmailArgs, name="send_email", side_effect=SideEffect.WRITE, timeout_s=10)
def send_email(args: EmailArgs) -> str:
    """Send an email. (Demo only: nothing leaves this machine.)"""
    if not args.api_key.startswith("sk-"):
        raise ToolError("The API key was rejected.")
    return f"Queued email to {args.to}: {args.subject!r}"
