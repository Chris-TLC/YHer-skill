"""Strict HTTP request schemas for the canonical Demo API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class StartSessionRequest(StrictRequest):
    user_id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")
    node: str = Field(min_length=1, max_length=120)
    budget_tier: Literal["30min", "1h", "2h", "3h+"] = "30min"
    grade: str = Field(default="高二", min_length=1, max_length=20)
    learning_purpose: str = Field(default="review", min_length=1, max_length=40)


class SubmitAnswerRequest(StrictRequest):
    assignment_id: str = Field(min_length=1, max_length=128)
    submission_id: str = Field(min_length=1, max_length=128)
    answer: str = Field(max_length=20_000)


class LearningAckRequest(StrictRequest):
    action_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")


class WatchRequest(StrictRequest):
    rec_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    watch_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    watched_seconds: float = Field(default=0.0, ge=0.0, le=86_400.0)
    completed: bool = False
