# True Inflation Canada Walkthrough (Economist-Focused)

## Purpose
True Inflation Canada is an experimental, open-source nowcast that estimates Canadian inflation in near real time using public data sources.
It is not an official Statistics Canada CPI release.

## Core Definitions
- YoY inflation: percent change versus the same month 12 months earlier; less volatile than month-over-month readings.
- Nowcast: model-based estimate generated before official publication.
- Deviation from Expectations: nowcast YoY minus consensus YoY, when consensus data is available.
- MAE (Mean Absolute Error): average absolute gap between nowcast and official outcomes over an evaluation window.

## Data Architecture
- Official CPI history comes from Statistics Canada and is stored for context and denominator calculations.
- Live nowcast history is authentic only: no synthetic backfilled nowcast points are written for pre-live dates.
- Source health and gate status determine publishability and confidence.

## YoY Projection Logic
The model first computes an MoM proxy from weighted category signals, then projects a current-month index and converts that to YoY.

Projection steps:
1. Use latest official index for the prior month (`I_t-1`).
2. Apply month-to-date prorating to MoM signal.
3. Compute projected current-month index.
4. Compare to official index from the same month one year earlier (`I_t-12`).

## Startup Phase (First Weeks)
- Live tracking begins February 16, 2026.
- The green nowcast line appears short initially by design.
- The gold official YoY line remains full history.
- Category Contribution Ranking requires at least two consecutive live runs to show stable deltas.
- MAE and directional metrics are low-confidence until roughly 30 to 60 live days accumulate.

## Reading the Dashboard
- Main chart: Nowcast vs Official CPI (YoY).
- Consensus card: may show unavailable when no robust aggregated forecast is present.
- Deviation from Expectations: shown only when consensus exists.
- Coverage Ratio: share of CPI basket with usable data in the run.
- Basket Weights: fixed CPI component weights used to aggregate category signals.

## Transparency and Limits
- Public-source feeds can be delayed, sparse, or revised.
- Signals are filtered and quality-scored, but this is still a research system.
- Compatibility fields remain temporarily available (`nowcast_mom_pct`, `consensus_spread_yoy`) during migration.

## Reference
Statistics Canada CPI program and methodology:
- https://www.statcan.gc.ca/en/statistical-programs/document/2301_D2_V4
