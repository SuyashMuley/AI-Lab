"""The README is executable documentation: its runnable blocks run, and print what it says."""

import contextlib
import io
import re
from pathlib import Path

from typer.testing import CliRunner

from toolrouter import cli

README = Path(__file__).parents[2] / "README.md"
FENCE = re.compile(r"^```(\w*)\n(.*?)^```", re.S | re.M)
runner = CliRunner()


def blocks() -> list[tuple[str, str]]:
    return FENCE.findall(README.read_text())


def test_runnable_python_blocks_run_and_print_exactly_what_the_readme_shows() -> None:
    namespace: dict[str, object] = {}
    found = blocks()
    ran = 0
    for index, (lang, body) in enumerate(found):
        if lang != "python" or not body.startswith("# readme-test"):
            continue
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(compile(body, f"<README block {index}>", "exec"), namespace)
        ran += 1
        assert index + 1 < len(found) and found[index + 1][0] == "text", (
            f"README block {index} has no output block after it"
        )
        assert out.getvalue().strip() == found[index + 1][1].strip(), f"README block {index}"
    assert ran >= 3  # the quick start, policy/hooks/secrets, and the provider-messages block


def help_text(*args: str) -> str:
    result = runner.invoke(cli.app, [*args, "--help"], terminal_width=200)
    assert result.exit_code == 0
    return result.stdout


def test_every_cli_command_and_flag_the_readme_mentions_exists() -> None:
    text = README.read_text()
    all_help = " ".join(help_text(*c) for c in ([], ["list"], ["schema"], ["call"], ["chat"]))
    for command in ("list", "schema", "call", "chat"):
        assert command in help_text()
    cli_section = text[text.index("## The `toolrouter` CLI") : text.index("## Guarantees")]
    for flag in set(re.findall(r"`?(--[a-z][a-z-]+)", cli_section)):
        assert flag in all_help, f"README mentions {flag}, which the CLI does not have"
    for fmt in cli.SchemaFormat:
        assert fmt.value in cli_section, f"README does not mention --format {fmt.value}"


def test_documented_error_codes_and_defaults_are_real() -> None:
    import inspect

    from toolrouter import DEFAULT_LIMITS, ErrorCode, ToolRouter, tool

    text = README.read_text()
    for code in ErrorCode:
        assert f"`{code.value}`" in text, f"error code {code.value} is undocumented"
    assert (DEFAULT_LIMITS.max_bytes, DEFAULT_LIMITS.max_depth) == (64 * 1024, 32)
    params = inspect.signature(ToolRouter).parameters
    assert params["max_threads"].default == 32
    assert params["max_concurrency"].default == 4
    t = tool(name="t", description="d")(lambda: None)
    assert (t.timeout_s, t.max_output_chars) == (30.0, 20_000)
    for phrase in ("default 30", "default 32", "20,000", "64 KiB", "depth 32"):
        assert phrase in text, phrase
