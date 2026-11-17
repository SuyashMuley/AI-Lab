"""Budgets: every run is bounded. More axes arrive with the milestones that enforce them."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Budgets(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_steps: int = Field(default=12, ge=1, description="Model calls per run.")
    max_tool_calls: int = Field(default=30, ge=0, description="Tool calls per run (0 = no tools).")
    model_timeout_s: float = Field(default=120.0, gt=0, allow_inf_nan=False)
