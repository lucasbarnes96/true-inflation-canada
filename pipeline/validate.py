from __future__ import annotations

from typing import Callable

from .types import ValidationResult


def build_validation_result(
    *,
    qa_checks: list[dict],
    as_of_day: str,
    representativeness_ratio: float,
    reconciliation: dict,
    imputation: dict,
    source_sla_days: dict[str, int],
    computed_health: list[dict],
    retry_offsets_minutes: list[int],
    build_qa_failure_fingerprint_fn: Callable[[list[dict]], dict],
    check_pass_rate_fn: Callable[[list[dict], str], float | None],
    source_pass_rate_30d_fn: Callable[[str], dict[str, float]],
    source_freshness_rate_30d_fn: Callable[[str], dict[str, float]],
    round_or_none_fn: Callable[[float | None, int], float | None],
) -> ValidationResult:
    this_run_contract_pass_rate = (
        round_or_none_fn(sum(1 for c in qa_checks if c.get("passed")) / len(qa_checks), 4)
        if qa_checks
        else 0.0
    )
    this_run_freshness_pass_rate = check_pass_rate_fn(qa_checks, "freshness")
    if this_run_freshness_pass_rate is None:
        this_run_freshness_pass_rate = 0.0

    trailing_by_source = source_pass_rate_30d_fn(as_of_day)
    trailing_freshness_by_source = source_freshness_rate_30d_fn(as_of_day)
    source_contract_pass_rate = (
        round_or_none_fn(sum(trailing_by_source.values()) / len(trailing_by_source), 4)
        if trailing_by_source
        else this_run_contract_pass_rate
    )
    source_freshness_pass_rate = (
        round_or_none_fn(sum(trailing_freshness_by_source.values()) / len(trailing_freshness_by_source), 4)
        if trailing_freshness_by_source
        else this_run_freshness_pass_rate
    )
    qa_failure_fingerprint = build_qa_failure_fingerprint_fn(qa_checks)

    expected_sources = set(source_sla_days.keys())
    observed_sources = {
        row.get("source")
        for row in computed_health
        if isinstance(row, dict) and isinstance(row.get("source"), str)
    }
    missing_sources = sorted(expected_sources - observed_sources)
    source_inventory_ratio = round_or_none_fn(
        (len(observed_sources) / len(expected_sources)) if expected_sources else 1.0,
        4,
    )

    qa_summary = {
        "this_run_source_contract_pass_rate": this_run_contract_pass_rate,
        "this_run_source_freshness_pass_rate": this_run_freshness_pass_rate,
        "trailing_30d_source_contract_pass_rate": source_contract_pass_rate,
        "trailing_30d_source_freshness_pass_rate": source_freshness_pass_rate,
        "source_contract_pass_rate": source_contract_pass_rate,
        "source_freshness_pass_rate": source_freshness_pass_rate,
        "fresh_weight_ratio": representativeness_ratio,
        "cross_source_disagreement_score": reconciliation["cross_source_disagreement_score"],
        "cross_source_disagreement_by_category": reconciliation["cross_source_disagreement_by_category"],
        "quarantine_sources": reconciliation["quarantine_sources"],
        "quarantine_reasons": reconciliation["quarantine_reasons"],
        "imputation_used": imputation["imputation_used"],
        "imputed_categories": imputation["imputed_categories"],
        "imputed_weight_ratio": imputation["imputed_weight_ratio"],
        "source_inventory_ratio": source_inventory_ratio,
        "expected_sources": sorted(expected_sources),
        "observed_sources": sorted(observed_sources),
        "missing_sources": missing_sources,
        "source_checks": qa_checks,
        "failure_fingerprint": qa_failure_fingerprint,
        "retry_schedule_minutes": retry_offsets_minutes,
    }
    return ValidationResult(
        this_run_source_contract_pass_rate=float(this_run_contract_pass_rate or 0.0),
        this_run_source_freshness_pass_rate=float(this_run_freshness_pass_rate or 0.0),
        trailing_30d_source_contract_pass_rate=float(source_contract_pass_rate or 0.0),
        trailing_30d_source_freshness_pass_rate=float(source_freshness_pass_rate or 0.0),
        qa_failure_fingerprint=qa_failure_fingerprint,
        qa_summary=qa_summary,
    )
