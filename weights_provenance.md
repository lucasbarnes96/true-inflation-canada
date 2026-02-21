# Weights Provenance (V1.3.1)

This document explains how tracked basket weights are selected for True Inflation Canada and how they map to Statistics Canada sources.

## Primary sources
- StatCan table: 18-10-0007-01 (basket weights)
  - https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1810000701
- StatCan 2025 basket analysis (based on 2024 expenditures): 62F0014M2025003
  - https://www150.statcan.gc.ca/n1/pub/62f0014m/62f0014m2025003-eng.htm

## Reference period
- Basket reference year: 2024 expenditures
- Effective month in CPI publication cycle: May 2025

## Tracked weights used by the model
- housing: 0.2941
- food: 0.1691
- transport: 0.1690
- recreation_education: 0.1012
- energy (proxy): 0.0800
- health_personal: 0.0505
- communication (proxy): 0.0350

Tracked share total: 0.8989 (89.89%)

## Omitted components in V1
- household_operations_furnishings_equipment (approx 0.1325)
- clothing_footwear (approx 0.0440)
- alcohol_tobacco_cannabis (approx 0.0400)

Rationale: these components are either difficult to scrape at high quality daily or less informative for high-frequency nowcast signal quality in V1.

## Mapping notes
- `communication` is represented as a project proxy mapped within broader official CPI structures.
- `energy` is treated as a high-signal proxy spanning transport/shelter-related cost pressure.

## Transparency commitment
Every snapshot includes `meta.weights` so third parties can audit model inputs and reproduce weighted aggregation assumptions.
