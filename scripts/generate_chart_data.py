import yfinance as yf
import pandas as pd
import requests
import json
import os
import urllib.request
import urllib.error
import zipfile
import io
import time
from datetime import datetime

# Bank of Canada Valet API Series IDs
ADJUSTERS_CONFIG = {
    'M1+': 'V41552785',
    'M1++': 'V41552788',
    'M3': 'V41552794',
    'CPI': 'V41690973' # CPI All-items
}

ASSETS_YAHOO_DIRECT = {
    'TSX': '^GSPTSE',
    'Canadian REITs': 'XRE.TO',
    'Bitcoin (CAD)': 'BTC-CAD',
    'Ethereum (CAD)': 'ETH-CAD',
    'Crude Oil': 'CL=F'
}

ASSETS_YAHOO_CALC_CAD = {
    'S&P 500 (CAD)': '^GSPC',
    'NASDAQ (CAD)': '^IXIC',
    'Dow Jones (CAD)': '^DJI',
    'Gold (CAD)': 'GC=F',
    'Silver (CAD)': 'SI=F'
}

def fetch_valet_series(series_id, retries=3, delay=2):
    url = f"https://www.bankofcanada.ca/valet/observations/{series_id}/json"
    for attempt in range(retries):
        try:
            res = requests.get(url, timeout=15)
            res.raise_for_status()
            data = res.json()
            records = []
            for obs in data['observations']:
                if series_id in obs:
                    records.append({
                        'Date': pd.to_datetime(obs['d']),
                        'Value': float(obs[series_id]['v'])
                    })
            df = pd.DataFrame(records).set_index('Date')
            return df.resample('ME').last().ffill()
        except requests.exceptions.RequestException as e:
            print(f"  Attempt {attempt + 1} failed for {url}: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
    raise Exception(f"Failed to fetch {url} after {retries} attempts")

def fetch_statcan_csv(url, filter_func, retries=3, delay=2):
    print(f"  Downloading StatCan {url}...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                with zipfile.ZipFile(io.BytesIO(response.read())) as z:
                    csv_filename = z.namelist()[0]
                    with z.open(csv_filename) as f:
                        df = pd.read_csv(f)
                        filtered = filter_func(df)
                        
                        # Format index to Date
                        if 'REF_DATE' in filtered.columns:
                            # some are YYYY, some are YYYY-MM
                            filtered['Date'] = pd.to_datetime(filtered['REF_DATE'], format="mixed", errors='coerce')
                            filtered = filtered.set_index('Date')
                            filtered = filtered[['VALUE']].rename(columns={'VALUE': 'Value'})
                            return filtered
            return pd.DataFrame()
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed for {url}: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
    print(f"  Final failure for StatCan {url}")
    return pd.DataFrame()

def fetch_yahoo_with_retry(ticker, retries=3, delay=2):
    for attempt in range(retries):
        try:
            t_obj = yf.Ticker(ticker)
            hist = t_obj.history(period="max")
            if not hist.empty:
                return hist
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed for {ticker}: {e}")
        if attempt < retries - 1:
            time.sleep(delay)
    return pd.DataFrame()

def generate_chart_data():
    output = {
        'metadata': {
            'description': 'Historical asset prices and inflation adjusters for True Inflation Canada',
            'updated_at': datetime.now().isoformat()
        },
        'adjusters': {},
        'assets': {}
    }

    print("Fetching Adjusters from Bank of Canada...")
    for name, s_id in ADJUSTERS_CONFIG.items():
        print(f"  Fetching {name}...")
        try:
            df = fetch_valet_series(s_id)
            first_val = df['Value'].dropna().iloc[0]
            output['adjusters'][name] = {
                'dates': df.index.strftime('%Y-%m').tolist(),
                'values': df['Value'].tolist(),
                'normalized': (df['Value'] / first_val).tolist()
            }
        except Exception as e:
            print(f"  Error fetching {name}: {e}")

    print("Fetching Assets from Yahoo Finance...")
    for name, ticker in ASSETS_YAHOO_DIRECT.items():
        print(f"  Fetching {name}...")
        try:
            hist = fetch_yahoo_with_retry(ticker)
            if hist.empty:
                continue
            hist_monthly = hist['Close'].resample('ME').last()
            hist_monthly.index = hist_monthly.index.tz_localize(None)
            
            output['assets'][name] = {
                'dates': hist_monthly.index.strftime('%Y-%m').tolist(),
                'values': hist_monthly.round(2).tolist()
            }
        except Exception as e:
            print(f"  Error fetching {name}: {e}")

    print("Fetching CAD exchange rate for calculations...")
    try:
        cad_hist = fetch_yahoo_with_retry("CAD=X")['Close'].resample('ME').last()
        cad_hist.index = cad_hist.index.tz_localize(None)
        
        for name, ticker in ASSETS_YAHOO_CALC_CAD.items():
            print(f"  Fetching {name}...")
            hist = fetch_yahoo_with_retry(ticker)['Close'].resample('ME').last()
            hist.index = hist.index.tz_localize(None)
            
            # Multiply by CAD=X
            combined = (hist * cad_hist).dropna()
            
            output['assets'][name] = {
                'dates': combined.index.strftime('%Y-%m').tolist(),
                'values': combined.round(2).tolist()
            }
    except Exception as e:
        print(f"  Error fetching Yahoo Calc: {e}")

    print("Fetching StatCan Data...")
    
    # 1. NHPI
    try:
        def filter_nhpi(df):
            return df[(df['GEO'] == 'Canada') & (df['New housing price indexes'] == 'Total (house and land)')]
        df_nhpi = fetch_statcan_csv("https://www150.statcan.gc.ca/n1/en/tbl/csv/18100205-eng.zip", filter_nhpi)
        if not df_nhpi.empty:
            df_monthly = df_nhpi.resample('ME').last().ffill()
            output['assets']['Canadian House Prices (NHPI)'] = {
                'dates': df_monthly.index.strftime('%Y-%m').tolist(),
                'values': df_monthly['Value'].round(2).tolist()
            }
    except Exception as e:
         print(f"  Error fetching NHPI: {e}")

    # 2. Labour Productivity
    try:
        def filter_prod(df):
            return df[(df['GEO'] == 'Canada') & (df['Sector'] == 'Business sector') & (df['Labour productivity measures and related measures'] == 'Labour productivity')]
        df_prod = fetch_statcan_csv("https://www150.statcan.gc.ca/n1/en/tbl/csv/36100206-eng.zip", filter_prod)
        if not df_prod.empty:
            df_monthly = df_prod.resample('ME').ffill() # forward fill quarters to months
            output['assets']['Labour Productivity'] = {
                'dates': df_monthly.index.strftime('%Y-%m').tolist(),
                'values': df_monthly['Value'].round(2).tolist()
            }
    except Exception as e:
         print(f"  Error fetching Labour Productivity: {e}")

    # 3. Median Income
    try:
        def filter_inc(df):
            return df[(df['GEO'] == 'Canada') & (df['Income concept'] == 'Median market income') & (df['Economic family type'] == 'Economic families and persons not in an economic family')]
        df_inc = fetch_statcan_csv("https://www150.statcan.gc.ca/n1/en/tbl/csv/11100190-eng.zip", filter_inc)
        if not df_inc.empty:
            # Income is annual, forward fill to months
            df_monthly = df_inc.resample('ME').ffill()
            output['assets']['Median Household Income'] = {
                'dates': df_monthly.index.strftime('%Y-%m').tolist(),
                'values': df_monthly['Value'].round(2).tolist()
            }
    except Exception as e:
         print(f"  Error fetching Median Income: {e}")


    print("Saving to data/chart_data.json...")
    os.makedirs('data', exist_ok=True)
    with open('data/chart_data.json', 'w') as f:
        json.dump(output, f)
    print("Execution complete!")

if __name__ == "__main__":
    generate_chart_data()
