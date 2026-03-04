from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

from .types import CollectedRunData


def collect_run_data(
    *,
    historical: dict,
    now: datetime,
    source_contracts: dict[str, dict],
    collect_all_quotes_fn: Callable[..., tuple[list[Any], list[Any], dict, list[dict]]],
    recompute_source_health_fn: Callable[..., list[dict]],
) -> CollectedRunData:
    quotes, source_health, collection_diagnostics, qa_checks = collect_all_quotes_fn(
        historical=historical,
        now=now,
        source_contracts=source_contracts,
    )
    computed_source_health = recompute_source_health_fn(source_health, now)
    return CollectedRunData(
        quotes=quotes,
        source_health=source_health,
        computed_source_health=computed_source_health,
        collection_diagnostics=collection_diagnostics,
        qa_checks=qa_checks,
    )
