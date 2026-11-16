import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from toolrouter import ReadOnlyPolicy, ToolError, ToolRegistry, ToolRouter, tool
from toolrouter.demo.chat import ChatReport, ModelTimeoutError, run_chat
from toolrouter.demo.tools import calculator, make_file_tools


def call(name: str, **arguments: Any) -> dict[str, Any]:
    return {"function": {"name": name, "arguments": arguments}}


def says(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text}


def asks(*calls: dict[str, Any], content: str = "") -> dict[str, Any]:
    return {"role": "assistant", "content": content, "tool_calls": list(calls)}


class Script:
    """A scripted model: returns the next prepared message and records what it was shown."""

    def __init__(self, *messages: Any) -> None:
        self.messages = list(messages)
        self.seen: list[list[Any]] = []
        self.tools_seen: list[list[dict[str, Any]]] = []

    async def __call__(self, messages: list[Any], tools: list[dict[str, Any]]) -> Any:
        self.seen.append(list(messages))
        self.tools_seen.append(tools)
        return self.messages.pop(0)


def router(*tools: Any, **kw: Any) -> ToolRouter:
    return ToolRouter(ToolRegistry(tools or [calculator]), **kw)


async def chat(script: Script, r: ToolRouter | None = None, **kw: Any) -> ChatReport:
    return await run_chat("question", router=r or router(), chat_fn=script, **kw)


def roles(report: ChatReport) -> list[str]:
    return [m["role"] if isinstance(m, dict) else m.role for m in report.messages]


# ---- the happy paths ----


async def test_answer_without_tools() -> None:
    script = Script(says("42"))
    report = await chat(script)
    assert report.ok
    assert report.answer == "42"
    assert (report.turns, report.tool_calls, report.repair_turns) == (1, 0, 0)
    assert roles(report) == ["system", "user", "assistant"]
    assert [t["function"]["name"] for t in script.tools_seen[0]] == ["calculator"]


async def test_one_tool_call_then_an_answer() -> None:
    script = Script(asks(call("calculator", expression="17 * 23 + 4")), says("It is 395."))
    report = await chat(script)
    assert report.ok
    assert report.answer == "It is 395."
    assert (report.turns, report.tool_calls) == (2, 1)
    assert roles(report) == ["system", "user", "assistant", "tool", "assistant"]
    tool_message = report.messages[3]
    assert tool_message == {"role": "tool", "tool_name": "calculator", "content": "395"}
    # the model saw its own tool result on the second turn
    assert script.seen[1][-1] == tool_message


async def test_parallel_calls_come_back_in_order() -> None:
    script = Script(
        asks(
            call("calculator", expression="1+1"),
            call("calculator", expression="2+2"),
            call("calculator", expression="3+3"),
        ),
        says("done"),
    )
    report = await chat(script)
    assert [m["content"] for m in report.messages if m["role"] == "tool"] == ["2", "4", "6"]
    assert report.tool_calls == 3


async def test_system_prompt_options() -> None:
    assert roles(await chat(Script(says("x")), system=None)) == ["user", "assistant"]
    report = await chat(Script(says("x")), system="Be terse.")
    assert report.messages[0] == {"role": "system", "content": "Be terse."}


async def test_sdk_style_message_objects_work() -> None:
    reply = SimpleNamespace(
        role="assistant",
        content="",
        tool_calls=[
            SimpleNamespace(
                function=SimpleNamespace(name="calculator", arguments={"expression": "2+3"})
            )
        ],
    )
    final = SimpleNamespace(role="assistant", content="five", tool_calls=None)
    report = await chat(Script(reply, final))
    assert report.answer == "five"
    assert report.messages[3]["content"] == "5"


async def test_missing_content_gives_an_empty_answer() -> None:
    report = await chat(Script(SimpleNamespace(role="assistant", content=None, tool_calls=None)))
    assert report.ok
    assert report.answer == ""


# ---- repair ----


async def test_a_model_repairs_a_bad_call_from_the_error_text() -> None:
    script = Script(
        asks(call("calculator", expr="2+2")),  # wrong argument name
        asks(call("calculator", expression="2+2")),  # fixed
        says("4"),
    )
    report = await chat(script)
    assert report.ok
    assert report.answer == "4"
    assert (report.turns, report.tool_calls, report.repair_turns) == (3, 2, 1)
    assert report.failures == {"arguments_invalid": 1}
    error_text = script.seen[1][-1]["content"]  # what the model was shown before retrying
    assert error_text.startswith("Error (arguments_invalid):")
    assert "expression: Field required" in error_text
    assert "Expected: calculator(expression: string" in error_text


async def test_an_unknown_tool_is_repairable_and_suggests_the_right_name() -> None:
    script = Script(
        asks(call("calculater", expression="1+1")),
        asks(call("calculator", expression="1+1")),
        says("2"),
    )
    report = await chat(script)
    assert report.ok
    assert report.repair_turns == 1
    assert "Did you mean 'calculator'?" in script.seen[1][-1]["content"]


async def test_gives_up_when_the_model_cannot_fix_its_calls() -> None:
    bad = asks(call("calculator", nonsense=1))
    script = Script(bad, bad, bad, says("never reached"))
    report = await chat(script, max_repairs=2)
    assert not report.ok
    assert report.answer is None
    assert report.gave_up == "could not fix its tool calls after 2 repair attempts"
    assert (report.turns, report.repair_turns) == (3, 3)
    assert (
        len(script.messages) == 1
    )  # the loop stopped; the last scripted reply was never asked for


async def test_zero_repairs_gives_up_at_the_first_bad_call() -> None:
    report = await chat(Script(asks(call("calculator", nonsense=1)), says("x")), max_repairs=0)
    assert report.gave_up is not None
    assert report.turns == 1


async def test_a_clean_turn_resets_the_repair_budget() -> None:
    bad = asks(call("calculator", nonsense=1))
    good = asks(call("calculator", expression="1+1"))
    # bad, good (reset), bad, bad -> with max_repairs=1 it gives up only on the 4th turn
    script = Script(bad, good, bad, bad, says("never"))
    report = await chat(script, max_repairs=1, max_turns=10)
    assert report.gave_up is not None
    assert (report.turns, report.repair_turns) == (4, 3)


async def test_failures_the_model_cannot_fix_do_not_use_the_repair_budget() -> None:
    @tool
    def flaky() -> str:
        """Always fails."""
        raise ToolError("backend down")

    script = Script(*[asks(call("flaky")) for _ in range(4)], says("sorry, it is down"))
    report = await chat(script, router(flaky), max_repairs=0, max_turns=10)
    assert report.ok  # it was never "stuck repairing"
    assert report.answer == "sorry, it is down"
    assert report.failures == {"tool_failed": 4}
    assert report.repair_turns == 0


async def test_max_turns_stops_an_endless_tool_loop() -> None:
    script = Script(*[asks(call("calculator", expression="1+1")) for _ in range(10)])
    report = await chat(script, max_turns=3)
    assert report.gave_up == "no final answer within 3 turns"
    assert report.turns == 3
    assert report.tool_calls == 3


async def test_failure_counts_are_aggregated_by_code() -> None:
    script = Script(
        asks(call("calculator", nonsense=1), call("nope"), call("calculator", expression="1/0")),
        says("ok"),
    )
    report = await chat(script)
    assert report.failures == {"arguments_invalid": 1, "unknown_tool": 1, "tool_failed": 1}


# ---- safety through the loop ----


async def test_a_read_only_policy_is_enforced_inside_the_loop(tmp_path: Path) -> None:
    write_file = [t for t in make_file_tools(tmp_path) if t.name == "write_file"][0]
    script = Script(
        asks(call("write_file", path="hi.txt", content="hello")),
        says("I could not write the file."),
    )
    report = await chat(script, router(write_file, policy=ReadOnlyPolicy()))
    assert report.ok
    assert report.failures == {"policy_denied": 1}
    assert report.repair_turns == 0  # a denial is not something the model can repair
    assert not (tmp_path / "hi.txt").exists()
    assert "not permitted" in script.seen[1][-1]["content"]


# ---- the model call itself ----


async def test_a_hung_model_call_times_out_instead_of_hanging() -> None:
    async def hung(messages: list[Any], tools: list[dict[str, Any]]) -> Any:
        await asyncio.sleep(30)

    start = time.perf_counter()
    with pytest.raises(ModelTimeoutError, match="0.05s"):
        await run_chat("q", router=router(), chat_fn=hung, model_timeout_s=0.05)
    assert time.perf_counter() - start < 1.0


async def test_the_model_timeout_applies_to_every_turn() -> None:
    state = {"n": 0}

    async def second_call_hangs(messages: list[Any], tools: list[dict[str, Any]]) -> Any:
        state["n"] += 1
        if state["n"] == 1:
            return asks(call("calculator", expression="1+1"))
        await asyncio.sleep(30)

    with pytest.raises(ModelTimeoutError):
        await run_chat("q", router=router(), chat_fn=second_call_hangs, model_timeout_s=0.05)
    assert state["n"] == 2


async def test_a_model_functions_own_timeout_error_is_not_our_timeout() -> None:
    async def own_timeout(messages: list[Any], tools: list[dict[str, Any]]) -> Any:
        raise TimeoutError("socket timed out")

    with pytest.raises(TimeoutError, match="socket") as ei:
        await run_chat("q", router=router(), chat_fn=own_timeout)
    assert not isinstance(ei.value, ModelTimeoutError)


async def test_timeout_can_be_disabled() -> None:
    async def slowish(messages: list[Any], tools: list[dict[str, Any]]) -> Any:
        await asyncio.sleep(0.1)
        return says("fine")

    report = await run_chat("q", router=router(), chat_fn=slowish, model_timeout_s=None)
    assert report.answer == "fine"


async def test_errors_from_the_model_function_propagate() -> None:
    async def broken(messages: list[Any], tools: list[dict[str, Any]]) -> Any:
        raise ConnectionError("ollama is down")

    with pytest.raises(ConnectionError):
        await run_chat("q", router=router(), chat_fn=broken)
