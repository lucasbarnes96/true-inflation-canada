from __future__ import annotations

from datetime import datetime
from typing import Callable

from .types import GateDecision


def evaluate_gate_decision(
    *,
    snapshot: dict,
    now: datetime,
    build_gate_diagnostics_fn: Callable[[dict], dict],
    evaluate_gate_fn: Callable[[dict], list[str]],
    validate_snapshot_fn: Callable[[dict], list[str]],
) -> tuple[GateDecision, dict]:
    gate_diagnostics = build_gate_diagnostics_fn(snapshot)
    blocked_conditions = evaluate_gate_fn(snapshot)
    blocked_conditions.extend(validate_snapshot_fn(snapshot))

    status = "published" if not blocked_conditions else "failed_gate"
    qa_status = "passed" if status == "published" else "failed"
    quality_tier = "good" if status == "published" else "blocked"
    publish_warning = None if status == "published" else "Gate blocked publication for this run."
    published_at = now.isoformat() if status == "published" else None
    return (
        GateDecision(
            blocked_conditions=blocked_conditions,
            status=status,
            qa_status=qa_status,
            quality_tier=quality_tier,
            publish_warning=publish_warning,
            published_at=published_at,
            execution_outcome="success",
            publication_outcome=status,
        ),
        gate_diagnostics,
    )
