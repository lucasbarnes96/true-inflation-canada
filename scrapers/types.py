from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass
class Quote:
    category: str
    item_id: str
    value: float
    observed_at: date
    source: str
    source_run_id: str | None = None


@dataclass
class SourceHealth:
    source: str
    category: str
    tier: int
    status: str
    last_success_timestamp: str | None
    detail: str
    source_run_id: str | None = None
    last_observation_period: str | None = None
