# True Inflation Canada

A reliable, open-source macroeconomic dashboard tracking the true value of the Canadian Dollar and various assets by adjusting for M3 money supply and the official Consumer Price Index. The project focuses on explicit sourcing, clear visual tracking, and lean pipeline auto-pilot.

Experimental open-source project using public data. Not an official StatCan CPI release.

## Architecture
- **No Backend**: The app is served as a fully static HTML/JS site, utilizing GitHub Pages or Vercel. 
- **Auto-Pilot Updates**: A daily GitHub Action runs a Python ingestion script (`scripts/generate_chart_data.py`) at 02:00 UTC. It fetches current macro data from the Bank of Canada, Yahoo Finance, and Statistics Canada, compiles it into a static JSON payload, and commits it directly back to the repository.
- **Dynamic Frontend**: The browser fetches the JSON payload on load and renders interactive charts and dynamic headlines without the need for a live API or database.

## Running Locally

1. **Install dependencies**
Ensure you have Python 3.11 installed.
```bash
pip install -r requirements.txt
```

2. **Generate the payload**
This fetches the latest data and writes to `data/chart_data.json`.
```bash
python scripts/generate_chart_data.py
```

3. **Serve the UI**
Serve the directory locally.
```bash
python -m http.server 8000
```
Open `http://localhost:8000`.

## License
MIT (`LICENSE`)

## Disclaimer
This project is not an official CPI pipeline and should not be used for actual monetary policy decisions.
