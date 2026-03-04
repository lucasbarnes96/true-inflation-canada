from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class CollectedRunData:
    quotes: list[Any]
    source_health: list[Any]
    computed_source_health: list[dict]
    collection_diagnostics: dict
    qa_checks: list[dict]


@dataclass
class ValidationResult:
    this_run_source_contract_pass_rate: float
    this_run_source_freshness_pass_rate: float
    trailing_30d_source_contract_pass_rate: float
    trailing_30d_source_freshness_pass_rate: float
    qa_failure_fingerprint: dict
    qa_summary: dict


@dataclass
class GateDecision:
    blocked_conditions: list[str]
    status: str
    qa_status: str
    quality_tier: str
    publish_warning: str | None
    published_at: str | None
    execution_outcome: str
    publication_outcome: str


@dataclass
class PersistenceInputs:
    snapshot: dict
    historical: dict
    run_path: Path
    status: str
    release_created_at: datetime | str
