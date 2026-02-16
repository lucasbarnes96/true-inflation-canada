from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


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
            "method_version": "v1.2.0",
            "window_days": window_days,
            "evaluated_points": 0,
            "mae_mom_pct": None,
            "directional_accuracy_pct": None,
            "lead_time_score_pct": None,
            "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        }

    days = sorted(historical.keys())[-window_days:]
    mae_terms: list[float] = []
    directional_hits = 0
    directional_total = 0
    lead_hits = 0
    lead_total = 0

    for day in days:
        payload = historical.get(day, {})
        headline = payload.get("headline", {})
        official = payload.get("official_cpi", {})

        nowcast = headline.get("nowcast_mom_pct")
        official_mom = official.get("mom_pct")
        divergence = headline.get("divergence_mom_pct")

        if divergence is None and nowcast is not None and official_mom is not None:
            divergence = float(nowcast) - float(official_mom)

        if divergence is not None:
            mae_terms.append(abs(float(divergence)))

        nowcast_sign = _sign(nowcast)
        official_sign = _sign(official_mom)
        if nowcast_sign is not None and official_sign is not None:
            directional_total += 1
            if nowcast_sign == official_sign:
                directional_hits += 1

        lead_signal = headline.get("lead_signal")
        if lead_signal is None:
            lead_signal = _lead_signal(nowcast)
        if official_sign is not None:
            lead_total += 1
            if (lead_signal == "up" and official_sign > 0) or (
                lead_signal == "down" and official_sign < 0
            ) or (lead_signal == "flat" and official_sign == 0):
                lead_hits += 1

    directional_accuracy = (directional_hits / directional_total * 100) if directional_total else None
    lead_score = (lead_hits / lead_total * 100) if lead_total else None
    mae = (sum(mae_terms) / len(mae_terms)) if mae_terms else None

    return {
        "method_version": "v1.2.0",
        "window_days": window_days,
        "evaluated_points": len(days),
        "mae_mom_pct": round(mae, 4) if mae is not None else None,
        "directional_accuracy_pct": round(directional_accuracy, 2) if directional_accuracy is not None else None,
        "lead_time_score_pct": round(lead_score, 2) if lead_score is not None else None,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }


def write_performance_summary(path: Path, historical: dict, window_days: int = 120) -> dict:
    summary = compute_performance_summary(historical=historical, window_days=window_days)
    path.write_text(json.dumps(summary, indent=2))
    return summary
