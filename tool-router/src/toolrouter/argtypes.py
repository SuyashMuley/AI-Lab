"""Strict argument types for tools.

`Confined` keeps a path argument inside a directory, enforced during validation:

    DataPath = Annotated[Path, Confined("/srv/data")]

    @tool(side_effect=SideEffect.READ)
    def read_file(path: DataPath) -> str: ...

It is a validation-time guard, not a sandbox: it cannot stop a symlink swapped in *after*
validation (real isolation arrives with sandboxed execution). The handler receives the
resolved real path, which narrows that window.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import GetCoreSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import PydanticCustomError, core_schema


@dataclass(frozen=True)
class Confined:
    """Annotated metadata: resolve a `Path` and require it to stay inside `root`.

    Relative paths are taken relative to `root`. `..` and symlinks are resolved first, so
    neither can escape. Use an absolute `root` (a relative one resolves against the cwd at
    validation time). Error messages never reveal the root's real location.
    """

    root: str | os.PathLike[str]
    must_exist: bool = False

    def __get_pydantic_core_schema__(
        self, source: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(self._check, handler(source))

    def _check(self, value: Path) -> Path:
        if "\x00" in str(value):
            raise PydanticCustomError("path_invalid", "path contains a null byte")
        root = Path(os.path.realpath(Path(self.root).expanduser()))
        candidate = value if value.is_absolute() else root / value
        try:
            real = Path(os.path.realpath(candidate))
        except (OSError, ValueError):
            raise PydanticCustomError("path_invalid", "path cannot be resolved") from None
        if not real.is_relative_to(root):
            raise PydanticCustomError("path_outside_root", "path is outside the allowed directory")
        if self.must_exist and not real.exists():
            raise PydanticCustomError("path_not_found", "path does not exist")
        return real


@dataclass(frozen=True)
class Secret:
    """Annotated metadata marking an argument as a secret:

        api_key: Annotated[str, Secret()]

    Secret values are redacted from hook events and logs and from the argument previews in
    validation errors, and a secret's default is never sent to a model. The handler still
    receives the plain value (use pydantic's `SecretStr` instead if you want it wrapped).
    `SecretStr`/`SecretBytes` fields are recognised automatically.
    """

    def __get_pydantic_json_schema__(
        self, schema: core_schema.CoreSchema, handler: Any
    ) -> JsonSchemaValue:
        return {**handler(schema), "writeOnly": True}
