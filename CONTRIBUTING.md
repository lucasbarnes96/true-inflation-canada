# Contributing to True Inflation Canada

## Scope
This project is exclusively focused on tracking macro-economic indicators using public, free data. Do not add paid or proprietary feeds. The project favors a lean architecture where data is aggregated into a static `chart_data.json` payload, rather than heavy databases or live web scrapers.

## Adding a Data Source
If you would like to track a new asset or a new macro-adjuster:

1. Open `scripts/generate_chart_data.py`.
2. For new assets (like Gold, Real Estate), add the ticker to `ASSETS_YAHOO_DIRECT` or `ASSETS_YAHOO_CALC_CAD` for localized pricing.
3. For official macro factors, add the series ID to `ADJUSTERS_CONFIG` (if hosted at the Bank of Canada) or build a resilient `fetch_statcan_csv` extraction block.
4. Ensure your addition implements robust retry parameters (`fetch_yahoo_with_retry` or similar) to ensure the GitHub Action auto-pilot does not become fragile.

## Testing Your Contribution
1. Run the aggregator script locally:
   ```bash
   python scripts/generate_chart_data.py
   ```
2. Verify that `data/chart_data.json` successfully rendered your new series.
3. Serve the site locally and verify that the UI gracefully accepts the new inputs via the chart dropdown selectors.

## Pull Request Notes
Every PR adding data sources should document:
- The exact public source URL.
- Rationale for inclusion.
- Ensuring the GitHub Action auto-pilot will not fail silently.
