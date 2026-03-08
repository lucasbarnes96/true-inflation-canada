import yfinance as yf
import pandas as pd
import requests
import json
import os
from datetime import datetime

# Bank of Canada Valet API Series IDs
# M1+: V41552785
# M1++: V41552788
# M2++: V41552792
# M3: V41552794
ADJUSTERS_CONFIG = {
    'M1+': 'V41552785',
    'M1++': 'V41552788',
    'M2++': 'V41552792',
    'M3': 'V41552794',
    'CPI': 'V41690973' # CPI All-items
}

ASSETS_CONFIG = {
    'TSX Composite': '^GSPTSE',
    'S&P 500': '^GSPC',
    'Canadian REITs': 'XRE.TO',
    'Gold (USD)': 'GC=F',
    'Bitcoin (USD)': 'BTC-USD',
    'Crude Oil': 'CL=F'
}

# TODO: Add StatCan Housing, Income, Food indices if possible via direct API/CSV
# For now, we'll start with these stable Yahoo/BoC ones.

def fetch_valet_series(series_id):
    url = f"https://www.bankofcanada.ca/valet/observations/{series_id}/json"
    res = requests.get(url)
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
            # Normalize to 1.0 at its own start for now (frontend will re-normalize as needed)
            first_val = df['Value'].iloc[0]
            output['adjusters'][name] = {
                'dates': df.index.strftime('%Y-%m').tolist(),
                'values': df['Value'].tolist(),
                'normalized': (df['Value'] / first_val).tolist()
            }
        except Exception as e:
            print(f"  Error fetching {name}: {e}")

    print("Fetching Assets from Yahoo Finance...")
    for name, ticker in ASSETS_CONFIG.items():
        print(f"  Fetching {name}...")
        try:
            t_obj = yf.Ticker(ticker)
            hist = t_obj.history(period="max")
            if hist.empty:
                continue
            hist_monthly = hist['Close'].resample('ME').last()
            hist_monthly.index = hist_monthly.index.tz_localize(None)
            
            output['assets'][name] = {
                'ticker': ticker,
                'dates': hist_monthly.index.strftime('%Y-%m').tolist(),
                'values': hist_monthly.round(2).tolist()
            }
        except Exception as e:
            print(f"  Error fetching {name}: {e}")

    print("Saving to data/chart_data.json...")
    os.makedirs('data', exist_ok=True)
    with open('data/chart_data.json', 'w') as f:
        json.dump(output, f)
    print("Execution complete!")

if __name__ == "__main__":
    generate_chart_data()
