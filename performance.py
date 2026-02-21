from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from gate_policy import METHOD_VERSION


def _sign(value: float | None, threshold: float = 0.02) -> int | None:
    if value is None:
        return None
    if value > threshold:
        return 1
    if value < -threshold:
        return -1
    return 0


def _lead_signal(nowcast: float | None) -> str:
    signal = _sign(nowcast)
    if signal is None:
        return "insufficient_data"
    if signal > 0:
        return "up"
    if signal < 0:
        return "down"
    return "flat"


def compute_performance_summary(historical: dict, window_days: int = 120) -> dict:
    if not historical:
        return {
            "method_version": METHOD_VERSION,
            "window_days": window_days,
            "evaluated_points": 0,
            "evaluated_live_points": 0,
            "mae_yoy_pct": None,
            "median_abs_error_yoy_pct": None,
            "directional_accuracy_yoy_pct": None,
            "bias_yoy_pct": None,
            "mae_mom_pct": None,
            "directional_accuracy_pct": None,
            "lead_time_score_pct": None,
            "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        }

    days = sorted(historical.keys())[-window_days:]
    mae_terms_mom: list[float] = []
    directional_hits_mom = 0
    directional_total_mom = 0
    lead_hits_mom = 0
    lead_total_mom = 0
    abs_errors_yoy: list[float] = []
    signed_errors_yoy: list[float] = []
    directional_hits_yoy = 0
    directional_total_yoy = 0
    evaluated_live_points = 0

    for day in days:
        payload = historical.get(day, {})
        if not isinstance(payload, dict):
            continue
        meta = payload.get("meta", {})
        if isinstance(meta, dict) and bool(meta.get("seeded")):
            continue

        evaluated_live_points += 1
        headline = payload.get("headline", {})
        official = payload.get("official_cpi", {})

        nowcast = headline.get("nowcast_mom_pct")
        official_mom = official.get("mom_pct")
        divergence = headline.get("divergence_mom_pct")

        if divergence is None and nowcast is not None and official_mom is not None:
            divergence = float(nowcast) - float(official_mom)

        if divergence is not None:
            mae_terms_mom.append(abs(float(divergence)))

        nowcast_sign = _sign(nowcast)
        official_sign = _sign(official_mom)
        if nowcast_sign is not None and official_sign is not None:
            directional_total_mom += 1
            if nowcast_sign == official_sign:
                directional_hits_mom += 1

        lead_signal = headline.get("lead_signal")
        if lead_signal is None:
            lead_signal = _lead_signal(nowcast)
        if official_sign is not None:
            lead_total_mom += 1
            if (lead_signal == "up" and official_sign > 0) or (
                lead_signal == "down" and official_sign < 0
            ) or (lead_signal == "flat" and official_sign == 0):
                lead_hits_mom += 1

        nowcast_yoy = headline.get("nowcast_yoy_pct")
        official_yoy = official.get("yoy_pct")
        if isinstance(nowcast_yoy, (int, float)) and isinstance(official_yoy, (int, float)):
            err = float(nowcast_yoy) - float(official_yoy)
            signed_errors_yoy.append(err)
            abs_errors_yoy.append(abs(err))
            nowcast_yoy_sign = _sign(float(nowcast_yoy), threshold=0.05)
            official_yoy_sign = _sign(float(official_yoy), threshold=0.05)
            if nowcast_yoy_sign is not None and official_yoy_sign is not None:
                directional_total_yoy += 1
                if nowcast_yoy_sign == official_yoy_sign:
                    directional_hits_yoy += 1

    directional_accuracy_mom = (directional_hits_mom / directional_total_mom * 100) if directional_total_mom else None
    lead_score_mom = (lead_hits_mom / lead_total_mom * 100) if lead_total_mom else None
    mae_mom = (sum(mae_terms_mom) / len(mae_terms_mom)) if mae_terms_mom else None
    directional_accuracy_yoy = (directional_hits_yoy / directional_total_yoy * 100) if directional_total_yoy else None
    mae_yoy = (sum(abs_errors_yoy) / len(abs_errors_yoy)) if abs_errors_yoy else None
    median_abs_yoy = statistics.median(abs_errors_yoy) if abs_errors_yoy else None
    bias_yoy = (sum(signed_errors_yoy) / len(signed_errors_yoy)) if signed_errors_yoy else None

    return {
        "method_version": METHOD_VERSION,
        "window_days": window_days,
        "evaluated_points": len(days),
        "evaluated_live_points": evaluated_live_points,
        "mae_yoy_pct": round(mae_yoy, 4) if mae_yoy is not None else None,
        "median_abs_error_yoy_pct": round(median_abs_yoy, 4) if median_abs_yoy is not None else None,
        "directional_accuracy_yoy_pct": round(directional_accuracy_yoy, 2) if directional_accuracy_yoy is not None else None,
        "bias_yoy_pct": round(bias_yoy, 4) if bias_yoy is not None else None,
        "mae_mom_pct": round(mae_mom, 4) if mae_mom is not None else None,
        "directional_accuracy_pct": round(directional_accuracy_mom, 2) if directional_accuracy_mom is not None else None,
        "lead_time_score_pct": round(lead_score_mom, 2) if lead_score_mom is not None else None,
        "compatibility_notes": [
            "YoY metrics are primary in v1.3.1.",
            "MoM metrics are retained for backward compatibility and should be treated as secondary.",
        ],
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }


def write_performance_summary(path: Path, historical: dict, window_days: int = 120) -> dict:
    summary = compute_performance_summary(historical=historical, window_days=window_days)
    path.write_text(json.dumps(summary, indent=2))
    return summary
