from __future__ import annotations

from typing import Any, Callable


def infer_execution_outcome(status: Any, current: Any) -> str:
    if isinstance(current, str) and current:
        return current
    if status in {"published", "failed_gate", "degraded_published"}:
        return "success"
    if status == "crashed":
        return "crash"
    return "unknown"


def infer_publication_outcome(status: Any, current: Any) -> str | None:
    if isinstance(current, str) and current:
        return current
    if status == "published":
        return "published"
    if status == "failed_gate":
        return "failed_gate"
    if status == "degraded_published":
        return "carry_forward"
    if status == "crashed":
        return "failed_gate"
    return status if isinstance(status, str) else None


def normalize_qa_summary(
    *,
    qa_summary: dict,
    meta: dict,
    reliability_rows: list[tuple[Any, ...]] | None = None,
    build_qa_failure_fingerprint_fn: Callable[[list[dict]], dict] | None = None,
) -> dict:
    summary_fingerprint = qa_summary.get("failure_fingerprint")
    if not isinstance(summary_fingerprint, dict):
        summary_fingerprint = {}

    meta_fingerprint = meta.get("qa_failure_fingerprint")
    if not isinstance(meta_fingerprint, dict):
        meta_fingerprint = {}

    checks = qa_summary.get("source_checks")
    if not isinstance(checks, list):
        checks = []
    checks = [row for row in checks if isinstance(row, dict)]

    failure_fingerprint = summary_fingerprint or meta_fingerprint
    if not failure_fingerprint and checks and build_qa_failure_fingerprint_fn is not None:
        failure_fingerprint = build_qa_failure_fingerprint_fn(checks)

    if failure_fingerprint:
        qa_summary["failure_fingerprint"] = failure_fingerprint
        meta["qa_failure_fingerprint"] = failure_fingerprint

    this_contract = qa_summary.get("this_run_source_contract_pass_rate")
    if this_contract is None:
        this_contract = qa_summary.get("source_contract_pass_rate")
    if this_contract is not None:
        qa_summary["this_run_source_contract_pass_rate"] = this_contract

    this_freshness = qa_summary.get("this_run_source_freshness_pass_rate")
    if this_freshness is None:
        this_freshness = qa_summary.get("source_freshness_pass_rate")
    if this_freshness is not None:
        qa_summary["this_run_source_freshness_pass_rate"] = this_freshness

    trailing_contract = qa_summary.get("trailing_30d_source_contract_pass_rate")
    trailing_freshness = qa_summary.get("trailing_30d_source_freshness_pass_rate")
    rows = reliability_rows or []
    contract_vals: list[float] = []
    freshness_vals: list[float] = []
    for row in rows:
        if len(row) >= 3:
            contract_vals.append(float(row[1]))
            freshness_vals.append(float(row[2]))
        elif len(row) == 2:
            contract_vals.append(float(row[0]))
            freshness_vals.append(float(row[1]))
    has_reliability = bool(contract_vals)
    if has_reliability:
        if trailing_contract is None:
            trailing_contract = round(sum(contract_vals) / len(contract_vals), 4)
        if trailing_freshness is None:
            trailing_freshness = round(sum(freshness_vals) / len(freshness_vals), 4)
    if trailing_contract is None:
        trailing_contract = this_contract
    if trailing_freshness is None:
        trailing_freshness = this_freshness
    if trailing_contract is not None:
        qa_summary["trailing_30d_source_contract_pass_rate"] = trailing_contract
    if trailing_freshness is not None:
        qa_summary["trailing_30d_source_freshness_pass_rate"] = trailing_freshness
    return failure_fingerprint if isinstance(failure_fingerprint, dict) else {}


def normalize_snapshot_run_state(
    payload: dict,
    *,
    reliability_rows: list[tuple[Any, Any, Any]] | None = None,
    build_qa_failure_fingerprint_fn: Callable[[list[dict]], dict] | None = None,
) -> dict:
    release = payload.get("release")
    if not isinstance(release, dict):
        release = {}
        payload["release"] = release
    release["execution_outcome"] = infer_execution_outcome(release.get("status"), release.get("execution_outcome"))
    release["publication_outcome"] = infer_publication_outcome(
        release.get("status"),
        release.get("publication_outcome"),
    )

    meta = payload.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        payload["meta"] = meta
    qa_summary = meta.get("qa_summary")
    if not isinstance(qa_summary, dict):
        qa_summary = {}
        meta["qa_summary"] = qa_summary
    normalize_qa_summary(
        qa_summary=qa_summary,
        meta=meta,
        reliability_rows=reliability_rows,
        build_qa_failure_fingerprint_fn=build_qa_failure_fingerprint_fn,
    )
    return payload
