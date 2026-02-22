# True Inflation Canada

Real-time Canadian inflation nowcast with strict publish gates, explicit source run timestamps, release intelligence, and published performance metrics.
Experimental open-source nowcast using public data. Not official StatCan CPI.

## What changed
- API-first architecture (`FastAPI + Pydantic`).
- Strict release gate (runs can fail and stay unpublished).
- Food publishability uses resilience rules (fresh + usable source diversity), not APIFY-only dependence.
- APIFY retries are attempted automatically before final gate evaluation.
- Source health includes explicit age text (`updated X days ago`).
- Collection now records DNS preflight diagnostics and failure signatures (DNS vs timeout vs anti-bot) for faster outage triage.

## Runtime and dependencies
- Python `3.11` is required (`.python-version`).
- Install pinned dependencies:

```bash
pip install -r requirements.txt
```

Optional fully pinned install:

```bash
pip install -r requirements.lock
```

## Environment
Create `.env`:

```bash
APIFY_TOKEN=your_token
# Optional overrides
# APIFY_ACTOR_IDS=sunny_eternity/loblaws-grocery-scraper,ko_red/loblaws-grocery-scraper
# APIFY_CATEGORY_URL=https://www.realcanadiansuperstore.ca/food/dairy-eggs/c/28003
# APIFY_MAX_ITEMS=50
```

## Run ingestion

```bash
python3 process.py
```

Quick source/network diagnosis:

```bash
python3 scripts/diagnose_sources.py
```

Bootstrap history first (recommended for first-time setup):

```bash
python3.11 scripts/seed_history.py
python3.11 process.py
```

`scripts/seed_history.py` backfills the last 365 days with tagged official CPI baselines (`meta.seeded=true`) and does not create synthetic nowcast history.
This preserves an authentic live nowcast track record while keeping official CPI context available for YoY comparison.

Output artifacts:
- `data/latest.json` (latest run, includes failed gates)
- `data/published_latest.json` (last gate-passing run)
- `data/historical.json` (published history)
- `data/runs/*.json` (versioned run snapshots)
- `data/releases.db` (run metadata table)
- `data/qa_runs.db` (source QA checks + 30-day reliability rollups)
- `data/source_contracts.json` (source-level QA contracts)

Exit code:
- `0` when published
- `1` when gate fails

## Run API

```bash
uvicorn api.main:app --reload
```

Endpoints:
- `GET /v1/nowcast/latest`
- `GET /v1/nowcast/history?start=YYYY-MM-DD&end=YYYY-MM-DD`
- `GET /v1/sources/health`
- `GET /v1/releases/latest`
- `GET /v1/methodology`
- `GET /v1/performance/summary`
- `GET /v1/sources/catalog`
- `GET /v1/releases/upcoming`
- `GET /v1/consensus/latest`
- `GET /v1/forecast/next_release`
- `GET /v1/calibration/status`
- `GET /v1/qa/status`

## Dashboard
Serve static UI and point it to API:

```bash
python3 -m http.server
```

Open `http://localhost:8000`. The dashboard fetches from `/v1/...` and expects the API on the same origin/reverse proxy.

## Release gate policy
A run is blocked (`failed_gate`) if any condition fails:
1. Food resilience fails (minimum fresh + usable food source diversity).
2. Required sources missing (`statcan_cpi_csv`, `statcan_gas_csv`, and at least one energy source).
3. Snapshot schema validation fails.
4. Category point minimums fail.
5. Official CPI metadata missing (`latest_release_month`).
6. Representativeness ratio below 85% fresh basket coverage.
7. APIFY retry diagnostics are recorded; stale APIFY does not automatically block if food resilience still passes.
8. Trailing source contract pass rate falls below policy threshold.
9. Imputed basket share exceeds policy threshold.
10. Cross-source disagreement exceeds per-category thresholds.

## Methodology v1.3.0 confidence rubric
- Inputs: release gate status, weighted coverage ratio, anomaly counts, and source diversity.
- `high`: no gate failures, high coverage, low anomalies, and no diversity penalty.
- `medium`: adequate coverage with anomaly or diversity penalties.
- `low`: gate failure, or low weighted coverage.

Additional headline and metadata fields:
- `headline.signal_quality_score` (0-100)
- `headline.lead_signal` (`up`, `down`, `flat`, `insufficient_data`)
- `headline.nowcast_yoy_pct` (primary nowcast metric)
- `headline.next_release_at_utc`
- `headline.consensus_yoy`
- `headline.deviation_yoy_pct`
- Compatibility fields (deprecated): `headline.nowcast_mom_pct`, `headline.consensus_spread_yoy`
- `meta.method_version` (`v1.3.0`)
- `meta.gate_diagnostics` (machine-readable gate checks)
- `meta.category_signal_inputs` (category source provenance with tier/freshness)
- `meta.forecast` (next-release forecast with confidence bounds, if eligible)
- `meta.calibration` (maturity tier and live calibration status)
- `meta.weights` (auditable StatCan-sourced weight provenance)

## Startup Phase (First Weeks)
- Live nowcast series begins on February 16, 2026.
- Green nowcast line is short initially by design; only authentic live runs are shown.
- Category Contribution Ranking needs at least 2 consecutive live runs for stable delta calculations.
- MAE and directional accuracy are low-confidence until roughly 30-60 live days accumulate.
- Forecast outputs may be withheld while calibration history is short.

## Glossary
- `YoY`: change versus the same month 12 months ago; typically less volatile than MoM.
- `Nowcast`: real-time estimate from scraped public signals before official release.
- `Deviation from Expectations`: `nowcast_yoy_pct - consensus_yoy` when consensus is available.
- `MAE`: Mean Absolute Error versus official values on the tracked evaluation window.

StatCan CPI methodology and basket references:
- https://www.statcan.gc.ca/en/statistical-programs/document/2301_D2_V4
- https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1810000701
- https://www150.statcan.gc.ca/n1/pub/62f0014m/62f0014m2025003-eng.htm

## Changelog
- `v1.3.1` - Alignment hardening: StatCan-linked weights provenance, YoY-first performance metrics, explicit calibration status, and public calibration endpoint.

## CI
GitHub Actions (`.github/workflows/scrape.yml`):
- Uses Python 3.11.
- Runs ingestion + tests.
- Enforces gate with `scripts/check_release_gate.py`.
- Commits generated data for all runs with explicit status flags (`published` vs `failed_gate`) so diagnostics remain fully auditable.

## Notes
This remains an experimental nowcast and is not an official CPI release.
