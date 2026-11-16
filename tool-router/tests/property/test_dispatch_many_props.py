"""A batch must give exactly the answers sequential dispatch gives, whatever the concurrency."""

import asyncio
import json
from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from tests.fakes import router_tools as rt
from tests.fuzz import fuzz
from toolrouter import ToolCall, ToolRegistry, ToolResult, ToolRouter

REGISTRY = ToolRegistry([*rt.SAFE, rt.delete])  # includes one WRITE tool

names = st.sampled_from(REGISTRY.names() + ["no_such_tool"])
arguments = st.one_of(
    st.just("{}"),
    st.just('{"text": "hi"}'),
    st.just('{"a": 1, "b": 2}'),
    st.just('{"n": 3}'),
    st.just('{"city": "Oslo"}'),
    st.just('{"flag": true}'),
    st.just('{"x": 4}'),
    st.just('{"a": "x"}'),  # invalid
    st.just("{"),  # invalid JSON
    st.dictionaries(st.text(max_size=4), st.integers(), max_size=3),
)
calls_strategy = st.lists(st.tuples(names, arguments), max_size=25)


def key(r: ToolResult) -> tuple[Any, ...]:
    """Everything except timing and the (non-comparable) underlying exception."""
    return (r.call_id, r.name, r.ok, r.content, r.truncated, r.error)


@fuzz(200)
@given(calls_strategy, st.integers(min_value=1, max_value=8), st.booleans())
def test_batch_equals_sequential_dispatch(
    spec: list[tuple[str, Any]], concurrency: int, serialize: bool
) -> None:
    calls = [
        ToolCall(id=f"c{i}", name=n, arguments=a if isinstance(a, str) else json.dumps(a))
        for i, (n, a) in enumerate(spec)
    ]

    async def go() -> tuple[list[ToolResult], list[ToolResult]]:
        router = ToolRouter(REGISTRY, max_concurrency=concurrency, serialize_writes=serialize)
        batch = await router.dispatch_many(calls)
        sequential = [await router.dispatch(call) for call in calls]
        return batch, sequential

    batch, sequential = asyncio.run(go())
    assert len(batch) == len(calls)
    assert [key(r) for r in batch] == [key(r) for r in sequential]
