"""Policies: allow or deny a call after its arguments are validated, before the handler runs.

A policy is a seam, not a security product: it decides from the tool, its side-effect level and
the *validated* arguments. Policies **fail closed**: if one raises, or returns anything other than
a PolicyDecision, the call is blocked. Keep policies fast; they run on the dispatch path.
"""

from __future__ import annotations

from collections.abc import Awaitable, Iterable
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel

from toolrouter.calls import ToolCall
from toolrouter.tool import SideEffect, Tool

_ORDER = {SideEffect.NONE: 0, SideEffect.READ: 1, SideEffect.WRITE: 2}


@dataclass(frozen=True)
class PolicyRequest:
    tool: Tool
    call: ToolCall  # the raw call (may contain secrets: do not log it)
    args: BaseModel  # validated arguments


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str = (
        ""  # shown to the model on denial: write it for the model, keep it free of internals
    )


ALLOW = PolicyDecision(True)


def deny(reason: str = "") -> PolicyDecision:
    return PolicyDecision(False, reason)


class Policy(Protocol):
    def check(self, request: PolicyRequest) -> PolicyDecision | Awaitable[PolicyDecision]: ...


class AllowAll:
    def check(self, request: PolicyRequest) -> PolicyDecision:
        return ALLOW


class MaxSideEffect:
    """Allow tools up to a side-effect level (NONE < READ < WRITE)."""

    def __init__(self, level: SideEffect) -> None:
        self.level = SideEffect(level)

    def check(self, request: PolicyRequest) -> PolicyDecision:
        effect = request.tool.side_effect
        if _ORDER[effect] <= _ORDER[self.level]:
            return ALLOW
        return deny(
            f"it has side effect {effect.value!r}, but only up to {self.level.value!r} is allowed"
        )


class ReadOnlyPolicy(MaxSideEffect):
    """Allow pure and read-only tools; deny anything that modifies external state."""

    def __init__(self) -> None:
        super().__init__(SideEffect.READ)

    def check(self, request: PolicyRequest) -> PolicyDecision:
        if request.tool.side_effect is SideEffect.WRITE:
            return deny("it modifies external state and this session is read-only")
        return ALLOW


class AllowList:
    """Allow only the named tools. A misspelled name denies (fails safe)."""

    def __init__(self, names: Iterable[str]) -> None:
        self.names = frozenset(names)

    def check(self, request: PolicyRequest) -> PolicyDecision:
        if request.tool.name in self.names:
            return ALLOW
        return deny("it is not on the list of allowed tools")


class DenyList:
    """Deny the named tools; allow the rest."""

    def __init__(self, names: Iterable[str]) -> None:
        self.names = frozenset(names)

    def check(self, request: PolicyRequest) -> PolicyDecision:
        if request.tool.name in self.names:
            return deny("it has been disabled")
        return ALLOW


class AllOf:
    """Every policy must allow. Evaluated in order; the first denial wins (and its reason)."""

    def __init__(self, *policies: Policy) -> None:
        self.policies = policies

    async def check(self, request: PolicyRequest) -> PolicyDecision:
        import inspect

        for policy in self.policies:
            outcome = policy.check(request)
            decision = await outcome if inspect.isawaitable(outcome) else outcome
            if not isinstance(decision, PolicyDecision):
                raise TypeError(f"{type(policy).__name__} returned {type(decision).__name__}")
            if not decision.allowed:
                return decision
        return ALLOW
