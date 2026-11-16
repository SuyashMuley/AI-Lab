import subprocess
import sys
from importlib import metadata
from pathlib import Path

from typer.testing import CliRunner

import toolrouter
from toolrouter import cli

PKG = Path(toolrouter.__file__).parent


def test_the_version_matches_the_installed_metadata() -> None:
    assert toolrouter.__version__ == metadata.version("toolrouter")


def test_version_flag() -> None:
    r = CliRunner().invoke(cli.app, ["--version"])
    assert r.exit_code == 0
    assert r.stdout.strip() == f"toolrouter {toolrouter.__version__}"


def test_every_public_name_exists_and_is_unique() -> None:
    assert len(toolrouter.__all__) == len(set(toolrouter.__all__))
    for name in toolrouter.__all__:
        assert hasattr(toolrouter, name), name
    assert not any(n.startswith("_") for n in toolrouter.__all__)


def test_the_package_is_marked_typed() -> None:
    assert (PKG / "py.typed").is_file()


def test_the_console_script_points_at_the_guarded_entry() -> None:
    scripts = {e.name: e.value for e in metadata.entry_points(group="console_scripts")}
    assert scripts["toolrouter"] == "toolrouter._entry:main"


def run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)


def test_a_missing_extra_gives_a_friendly_message_not_a_traceback() -> None:
    done = run(
        "import sys\n"
        "sys.modules['typer'] = None  # as if typer were not installed\n"
        "from toolrouter._entry import main\n"
        "main()\n"
    )
    assert done.returncode == 1
    assert "Traceback" not in done.stderr
    assert "pip install 'toolrouter[cli]'" in done.stderr
    assert "typer" in done.stderr


def test_other_import_errors_are_not_disguised() -> None:
    done = run(
        "import builtins\n"
        "real = builtins.__import__\n"
        "def fake(name, *a, **k):\n"
        "    if name == 'toolrouter.cli':\n"
        "        raise ModuleNotFoundError('unrelated breakage', name='something_else')\n"
        "    return real(name, *a, **k)\n"
        "builtins.__import__ = fake\n"
        "from toolrouter import _entry\n"
        "_entry.main()\n"
    )
    assert done.returncode != 0
    assert "pip install" not in done.stderr  # a real bug is not misreported as a missing extra
    assert "ModuleNotFoundError" in done.stderr
