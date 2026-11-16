"""Tiny Google-style docstring parser: a summary plus per-parameter descriptions."""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field

_SECTION = re.compile(
    r"^(Args|Arguments|Parameters|Params|Returns?|Yields?|Raises?|Examples?|Notes?|"
    r"Attributes|See Also|Warnings?):\s*$",
    re.IGNORECASE,
)
_PARAM = re.compile(r"^(\s+)\*{0,2}(\w+)(?:\s*\([^)]*\))?:\s*(.*)$")
_ARG_SECTIONS = {"args", "arguments", "parameters", "params"}


@dataclass(frozen=True)
class ParsedDoc:
    summary: str = ""
    params: dict[str, str] = field(default_factory=dict)


def _join_paragraphs(lines: list[str]) -> str:
    paragraphs: list[list[str]] = [[]]
    for line in lines:
        if line.strip():
            paragraphs[-1].append(line.strip())
        elif paragraphs[-1]:
            paragraphs.append([])
    return "\n\n".join(" ".join(p) for p in paragraphs if p)


def parse_docstring(doc: str | None) -> ParsedDoc:
    if not doc or not doc.strip():
        return ParsedDoc()
    lines = inspect.cleandoc(doc).splitlines()

    i = 0
    while i < len(lines) and not _SECTION.match(lines[i]):
        i += 1
    summary = _join_paragraphs(lines[:i])

    params: dict[str, list[str]] = {}
    while i < len(lines):
        header = _SECTION.match(lines[i])
        i += 1
        in_args = header is not None and header.group(1).lower() in _ARG_SECTIONS
        base_indent: int | None = None
        current: str | None = None
        while i < len(lines) and not _SECTION.match(lines[i]):
            line = lines[i]
            i += 1
            if not in_args or not line.strip():
                continue
            m = _PARAM.match(line)
            indent = len(line) - len(line.lstrip())
            if base_indent is None and m:
                base_indent = indent
            if m and indent == base_indent:
                current = m.group(2)
                params[current] = [m.group(3)]
            elif current is not None and base_indent is not None and indent > base_indent:
                params[current].append(line)
    return ParsedDoc(summary, {k: _join_paragraphs(v) for k, v in params.items()})
