from __future__ import annotations

from .types import GateDecision


def apply_release_decision(snapshot: dict, decision: GateDecision) -> dict:
    snapshot["release"]["status"] = decision.status
    snapshot["release"]["quality_tier"] = decision.quality_tier
    snapshot["release"]["qa_status"] = decision.qa_status
    snapshot["release"]["blocked_conditions"] = decision.blocked_conditions
    snapshot["release"]["lifecycle_states"].append(decision.status)
    snapshot["release"]["execution_outcome"] = decision.execution_outcome
    snapshot["release"]["publication_outcome"] = decision.publication_outcome
    snapshot["headline"]["publish_warning"] = decision.publish_warning
    if decision.published_at:
        snapshot["release"]["published_at"] = decision.published_at
    return snapshot
