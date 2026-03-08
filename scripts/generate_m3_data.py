import yfinance as yf
import pandas as pd
import requests
import json
import os

def generate_m3_data():
    print("Fetching Canadian M3 Money Supply from Valet API...")
    valet_url = "https://www.bankofcanada.ca/valet/observations/V41552794/json"
    res = requests.get(valet_url)
    res.raise_for_status()
    m3_data = res.json()

    m3_records = []
    for obs in m3_data['observations']:
        if 'V41552794' in obs:
            m3_records.append({
                'Date': pd.to_datetime(obs['d']),
                'M3': float(obs['V41552794']['v'])
            })

    m3_df = pd.DataFrame(m3_records).set_index('Date')
    # Resample to month-end to match financial data
    m3_df = m3_df.resample('ME').last().ffill()

    assets = {
        'TSX Composite': '^GSPTSE',
        'S&P 500': '^GSPC',
        'Canadian REITs': 'XRE.TO',
        'Gold (USD)': 'GC=F',
        'Bitcoin (USD)': 'BTC-USD',
        'Crude Oil': 'CL=F'
    }

    output = {
        'metadata': {
            'description': 'Historical asset prices normalized against Canadian M3 Money Supply'
        },
        'assets': {}
    }

    for name, ticker in assets.items():
        print(f"Fetching {name} ({ticker})...")
        try:
            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(period="max")
            
            if hist.empty:
                print(f"  Warning: No data for {name}")
                continue
            
            # Resample asset to monthly
            hist_monthly = hist['Close'].resample('ME').last()
            hist_monthly.index = hist_monthly.index.tz_localize(None)
            
            # Inner join with M3 data
            df_merged = pd.DataFrame({'Asset': hist_monthly}).join(m3_df, how='inner')
            df_merged.dropna(inplace=True)
            
            if df_merged.empty:
                print(f"  Warning: No overlapping dates for {name}")
                continue

            # Normalize M3 to 1.0 at the first available date for this specific asset
            # This ensures the Nominal and M3-Adjusted lines start at the exact same value point!
            first_m3 = df_merged['M3'].iloc[0]
            df_merged['M3_Normalized'] = df_merged['M3'] / first_m3
            
            # Calculated Adjusted Price
            df_merged['Adjusted'] = df_merged['Asset'] / df_merged['M3_Normalized']
            
            output['assets'][name] = {
                'ticker': ticker,
                'dates': df_merged.index.strftime('%Y-%m').tolist(),
                'nominal': df_merged['Asset'].round(2).tolist(),
                'm3_normalized': df_merged['M3_Normalized'].round(4).tolist(),
                'adjusted': df_merged['Adjusted'].round(2).tolist()
            }
            print(f"  Processed {len(df_merged)} months of data.")
        except Exception as e:
            print(f"  Error fetching {name}: {e}")

    print("Saving to data/history.json...")
    os.makedirs('data', exist_ok=True)
    with open('data/history.json', 'w') as f:
        json.dump(output, f)
    print("Generation complete!")

if __name__ == "__main__":
    generate_m3_data()
