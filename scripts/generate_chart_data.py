import io
import json
import time
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

from validate_chart_data import validate_payload

# Bank of Canada Valet API Series IDs
ADJUSTERS_CONFIG = {
    "M1+": "V41552785",
    "M1++": "V41552788",
    "M3": "V41552794",
    "CPI": "V41690973",  # CPI All-items
}

ASSETS_YAHOO_DIRECT = {
    "TSX": "^GSPTSE",
    "Canadian REITs": "XRE.TO",
    "Bitcoin (CAD)": "BTC-CAD",
    "Ethereum (CAD)": "ETH-CAD",
    "Crude Oil": "CL=F",
}

ASSETS_YAHOO_CALC_CAD = {
    "S&P 500 (CAD)": "^GSPC",
    "NASDAQ (CAD)": "^IXIC",
    "Dow Jones (CAD)": "^DJI",
    "Gold (CAD)": "GC=F",
    "Silver (CAD)": "SI=F",
}

STATCAN_ASSETS = (
    "Canadian House Prices (NHPI)",
    "Labour Productivity",
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "data" / "chart_data.json"


class DataFetchError(RuntimeError):
    pass


def fetch_valet_series(series_id, retries=3, delay=2):
    url = f"https://www.bankofcanada.ca/valet/observations/{series_id}/json"
    for attempt in range(retries):
        try:
            res = requests.get(url, timeout=15)
            res.raise_for_status()
            data = res.json()
            records = []
            for obs in data["observations"]:
                if series_id in obs:
                    records.append(
                        {
                            "Date": pd.to_datetime(obs["d"]),
                            "Value": float(obs[series_id]["v"]),
                        }
                    )
            if not records:
                raise DataFetchError(f"No observations returned for {series_id}")
            df = pd.DataFrame(records).set_index("Date").sort_index()
            return df.resample("ME").last().ffill()
        except requests.exceptions.RequestException as e:
            print(f"  Attempt {attempt + 1} failed for {url}: {e}")
        except (KeyError, ValueError, TypeError, DataFetchError) as e:
            raise DataFetchError(f"Invalid Bank of Canada payload for {series_id}: {e}") from e
        if attempt < retries - 1:
            time.sleep(delay)
    raise DataFetchError(f"Failed to fetch {url} after {retries} attempts")


def fetch_statcan_csv(url, filter_func, retries=3, delay=2):
    print(f"  Downloading StatCan {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                with zipfile.ZipFile(io.BytesIO(response.read())) as z:
                    csv_filename = z.namelist()[0]
                    with z.open(csv_filename) as f:
                        df = pd.read_csv(f)
                        filtered = filter_func(df)

            if filtered.empty:
                raise DataFetchError(f"Filter returned no rows for {url}")
            if "REF_DATE" not in filtered.columns or "VALUE" not in filtered.columns:
                raise DataFetchError(f"Missing REF_DATE/VALUE columns for {url}")

            filtered = filtered.copy()
            filtered["Date"] = pd.to_datetime(filtered["REF_DATE"], format="mixed", errors="coerce")
            filtered = filtered.dropna(subset=["Date"]).set_index("Date").sort_index()
            filtered = filtered[["VALUE"]].rename(columns={"VALUE": "Value"})
            if filtered.empty:
                raise DataFetchError(f"No dated rows remained after parsing {url}")
            return filtered
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed for {url}: {e}")
        if attempt < retries - 1:
            time.sleep(delay)
    raise DataFetchError(f"Failed to fetch StatCan dataset {url} after {retries} attempts")


def fetch_yahoo_with_retry(ticker, retries=3, delay=2):
    for attempt in range(retries):
        try:
            t_obj = yf.Ticker(ticker)
            hist = t_obj.history(period="max")
            if not hist.empty:
                return hist
            raise DataFetchError(f"Yahoo returned no rows for {ticker}")
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed for {ticker}: {e}")
        if attempt < retries - 1:
            time.sleep(delay)
    raise DataFetchError(f"Failed to fetch Yahoo Finance history for {ticker}")


def build_series_payload(series: pd.Series) -> dict:
    if series.empty:
        raise DataFetchError("Cannot build payload from an empty series")

    clean_series = series.dropna().sort_index()
    if clean_series.empty:
        raise DataFetchError("Cannot build payload from a fully null series")

    return {
        "dates": clean_series.index.strftime("%Y-%m").tolist(),
        "values": clean_series.round(2).tolist(),
    }


def build_adjuster_payload(df: pd.DataFrame) -> dict:
    value_series = df["Value"].dropna().sort_index()
    if value_series.empty:
        raise DataFetchError("Adjuster series is empty after dropping nulls")

    first_val = value_series.iloc[0]
    payload = build_series_payload(value_series)
    payload["normalized"] = (value_series / first_val).round(10).tolist()
    return payload


def build_output():
    output = {
        "metadata": {
            "description": "Historical asset prices and inflation adjusters for True Inflation Canada",
            "updated_at": datetime.now(UTC).isoformat(),
        },
        "adjusters": {},
        "assets": {},
    }

    print("Fetching Adjusters from Bank of Canada...")
    for name, s_id in ADJUSTERS_CONFIG.items():
        print(f"  Fetching {name}...")
        df = fetch_valet_series(s_id)
        output["adjusters"][name] = build_adjuster_payload(df)

    print("Fetching Assets from Yahoo Finance...")
    for name, ticker in ASSETS_YAHOO_DIRECT.items():
        print(f"  Fetching {name}...")
        hist = fetch_yahoo_with_retry(ticker)
        hist_monthly = hist["Close"].resample("ME").last()
        hist_monthly.index = hist_monthly.index.tz_localize(None)
        output["assets"][name] = build_series_payload(hist_monthly)

    print("Fetching CAD exchange rate for calculations...")
    cad_hist = fetch_yahoo_with_retry("CAD=X")["Close"].resample("ME").last()
    cad_hist.index = cad_hist.index.tz_localize(None)
    if cad_hist.empty:
        raise DataFetchError("CAD=X returned no rows")

    for name, ticker in ASSETS_YAHOO_CALC_CAD.items():
        print(f"  Fetching {name}...")
        hist = fetch_yahoo_with_retry(ticker)["Close"].resample("ME").last()
        hist.index = hist.index.tz_localize(None)

        combined = (hist * cad_hist).dropna()
        output["assets"][name] = build_series_payload(combined)

    print("Fetching StatCan Data...")

    def filter_nhpi(df):
        return df[
            (df["GEO"] == "Canada")
            & (df["New housing price indexes"] == "Total (house and land)")
        ]

    df_nhpi = fetch_statcan_csv(
        "https://www150.statcan.gc.ca/n1/en/tbl/csv/18100205-eng.zip",
        filter_nhpi,
    )
    df_monthly_nhpi = df_nhpi.resample("ME").last().ffill()
    output["assets"]["Canadian House Prices (NHPI)"] = build_series_payload(
        df_monthly_nhpi["Value"]
    )

    def filter_prod(df):
        return df[
            (df["GEO"] == "Canada")
            & (df["Sector"] == "Business sector")
            & (df["Labour productivity measures and related measures"] == "Labour productivity")
        ]

    df_prod = fetch_statcan_csv(
        "https://www150.statcan.gc.ca/n1/en/tbl/csv/36100206-eng.zip",
        filter_prod,
    )
    df_monthly_prod = df_prod.resample("ME").ffill()
    output["assets"]["Labour Productivity"] = build_series_payload(
        df_monthly_prod["Value"]
    )

    return output


def generate_chart_data():
    output = build_output()
    validation = validate_payload(output)
    print(
        "Validation summary: "
        f"required_adjusters={validation['required_adjusters_present']}/{validation['required_adjusters_expected']}, "
        f"required_assets={validation['required_assets_present']}/{validation['required_assets_expected']}, "
        f"latest_adjuster={validation['latest_dates']['adjusters']}, "
        f"latest_yahoo={validation['latest_dates']['yahoo_assets']}, "
        f"latest_statcan={validation['latest_dates']['statcan_assets']}, "
        f"publish_allowed={validation['publish_allowed']}"
    )

    print(f"Saving to {OUTPUT_PATH.relative_to(REPO_ROOT)}...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, sort_keys=True, separators=(",", ":"))
        f.write("\n")
    print("Execution complete!")
    return output


if __name__ == "__main__":
    generate_chart_data()
