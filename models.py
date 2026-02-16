from __future__ import annotations

from datetime import date, datetime
from typing import Literal

try:
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover - local fallback when deps are missing
    class BaseModel:  # type: ignore[override]
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        @classmethod
        def model_validate(cls, payload):
            if isinstance(payload, cls):
                return payload
            if isinstance(payload, dict):
                return cls(**payload)
            return payload

        def model_dump(self, mode: str | None = None):
            return self.__dict__

    def Field(default_factory=None, default=None):  # type: ignore[override]
        if default_factory is not None:
            return default_factory()
        return default


class QuoteRecord(BaseModel):
    category: str
    item_id: str
    value: float
    observed_at: date
    source: str
    source_run_id: str | None = None


class SourceHealthModel(BaseModel):
    source: str
    category: str
    tier: int
    status: Literal["fresh", "stale", "missing"]
    last_success_timestamp: str | None = None
    detail: str
    source_run_id: str | None = None
    last_observation_period: str | None = None
    age_days: int | None = None
    run_age_hours: float | None = None
    updated_days_ago: str = "unknown"


class CategorySummary(BaseModel):
    proxy_level: float | None = None
    daily_change_pct: float | None = None
    weight: float
    points: int = 0
    status: Literal["fresh", "stale", "missing"]


class ReleaseGateResult(BaseModel):
    passed: bool
    status: Literal["published", "failed_gate", "completed", "started"]
    blocked_conditions: list[str] = Field(default_factory=list)
    evaluated_at: datetime


class HeadlineModel(BaseModel):
    nowcast_mom_pct: float | None = None
    confidence: Literal["high", "medium", "low"]
    coverage_ratio: float
    signal_quality_score: int = 0
    lead_signal: Literal["up", "down", "flat", "insufficient_data"] = "insufficient_data"
    next_release_at_utc: datetime | None = None
    consensus_yoy: float | None = None
    consensus_spread_yoy: float | None = None
    method_label: str


class ReleaseMeta(BaseModel):
    run_id: str
    status: Literal["published", "failed_gate", "completed", "started"]
    lifecycle_states: list[str] = Field(default_factory=list)
    blocked_conditions: list[str] = Field(default_factory=list)
    created_at: datetime
    published_at: datetime | None = None


class NowcastSnapshot(BaseModel):
    as_of_date: date
    timestamp: datetime
    headline: HeadlineModel
    categories: dict[str, CategorySummary]
    official_cpi: dict
    bank_of_canada: dict
    source_health: list[SourceHealthModel]
    notes: list[str]
    meta: dict
    performance_ref: dict | None = None
    release: ReleaseMeta
