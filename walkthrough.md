# True Inflation Canada Walkthrough

## Purpose
True Inflation Canada tracks the true value of the Canadian Dollar and various assets (like the S&P 500, TSX, and Housing) by adjusting them against macro-economic parameters like Bank of Canada's M3 money supply and official Consumer Price Index. The project is designed to be lean, robust, and run flawlessly on auto-pilot.

## Architecture & Data Flow
1. **Data Ingestion Script (`scripts/generate_chart_data.py`)**:
   - The primary engine of the application.
   - It fetches real-time data from the Bank of Canada Valet API, Yahoo Finance, and Statistics Canada CSV zips.
   - It is hardened with robust retry logic (`fetch_yahoo_with_retry`, built-in delays) to prevent fragility due to temporary network timeouts or rate limits.
   - Generates a single compiled JSON payload: `data/chart_data.json`.

2. **Automated Autopilot (`.github/workflows/daily_chart_data.yml`)**:
   - Every day at 02:00 UTC, a GitHub Action automatically runs the `generate_chart_data.py` script.
   - If the script successfully fetches and parses new data, the GitHub Action automatically commits the updated `chart_data.json` back to the `main` branch.
   - You do not need to intervene.

3. **Frontend Application (`index.html`)**:
   - A static, vanilla HTML/JS setup customized for maximum aesthetics.
   - Fetches the compiled `data/chart_data.json`.
   - **Dynamic CPI Headline**: The headline CPI "2.3%" text box dynamically reads the latest Official CPI values from the dataset, calculates the exact Year-over-Year inflation percentage, calculates the monthly evaporation rate, and computes the inflation half-life automatically. The narrative adjusts as the data drops.
