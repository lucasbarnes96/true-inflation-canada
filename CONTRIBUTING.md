# Contributing to True Inflation Canada

## Scope
This project is free/public-data-only for the current roadmap. Do not add paid or proprietary feeds.

## Adding a Source
1. Add scraper module under `scrapers/`.
2. Return typed `Quote` and `SourceHealth` records.
3. Add source metadata to `source_catalog.py`.
4. Register scraper in `process.py` `SCRAPER_REGISTRY`.
5. Add/extend tests in `tests/`:
   - parser happy path
   - schema drift fallback
   - gate behavior if source is missing/stale

## Source Quality Requirements
- Include license and public/free proof URL.
- Prefer national coverage; if provincial-only, declare it in metadata.
- Define expected cadence and freshness SLA.
- Include actionable `detail` for failures.

## Validation Checklist
- `python -m unittest discover -s tests -p 'test_*.py'`
- `python process.py` completes and writes `data/latest.json`
- `api/main.py` endpoints still return valid JSON contracts

## Pull Request Notes
Every PR adding data sources should document:
- category mapping and rationale
- known regional biases
- fallback behavior when scraping fails

