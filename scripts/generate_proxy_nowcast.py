import yfinance as yf
import json
import datetime
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

TICKERS = {
    "cad_usd": "CADUSD=X",       # CAD to USD
    "gold": "GC=F",              # Gold Futures
    "btc": "BTC-USD",            # Bitcoin to USD
    "sp500": "^GSPC",            # S&P 500 Index
    "wheat": "ZW=F",             # Wheat Futures
    "oil": "CL=F",               # Crude Oil Futures
    "reit": "XRE.TO"             # iShares S&P/TSX Capped REIT Index
}

def fetch_latest_prices():
    prices = {}
    for name, ticker in TICKERS.items():
        try:
            # Fetch past 5 days to ensure we get a valid close even on weekends
            hist = yf.Ticker(ticker).history(period="5d")
            if not hist.empty:
                prices[name] = float(hist['Close'].iloc[-1])
            else:
                print(f"Warning: No data for {name} ({ticker})")
                prices[name] = None
        except Exception as e:
            print(f"Error fetching {name} ({ticker}): {e}")
            prices[name] = None
    return prices

def calculate_metrics(prices):
    # Base CAD Value in USD
    cad_usd = prices.get("cad_usd")
    
    metrics = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "raw_prices": prices,
        "debasement": {},
        "proxy_nowcast": {}
    }
    
    if not cad_usd:
        return metrics

    # --- Debasement Metrics ---
    # How much of an asset does 1 CAD buy?
    
    if prices.get("gold"):
        # Gold is priced in USD. 
        # 1 CAD buys (cad_usd) USD. 
        # (cad_usd) USD buys (cad_usd / gold_price_usd) ounces of gold.
        gold_oz_per_cad = cad_usd / prices["gold"]
        metrics["debasement"]["gold_oz_per_cad"] = gold_oz_per_cad
        
    if prices.get("btc"):
        btc_per_cad = cad_usd / prices["btc"]
        metrics["debasement"]["btc_per_cad"] = btc_per_cad
        
    if prices.get("sp500"):
        sp500_shares_per_cad = cad_usd / prices["sp500"]
        metrics["debasement"]["sp500_shares_per_cad"] = sp500_shares_per_cad

    # --- Proxy Nowcast Construction ---
    # This is a highly simplified, directional proxy using commodities and REITs.
    # To make it an "inflation" index, we look at the cost of these items in CAD.
    # We will compute a simple "Basket Value" in CAD.
    
    # Cost in CAD = USD Price / CADUSD
    try:
        oil_cad = prices["oil"] / cad_usd if prices.get("oil") else 0
        wheat_cad = prices["wheat"] / cad_usd if prices.get("wheat") else 0
        reit_cad = prices["reit"] # Already in CAD on TSX
        
        # Arbitrary proxy weighting for demonstration of a market-based cost basket.
        # e.g., 10 barrels of oil, 100 bushels of wheat, 100 shares of REIT
        proxy_basket_cad = (oil_cad * 10) + (wheat_cad * 1) + (reit_cad * 50)
        metrics["proxy_nowcast"]["basket_value_cad"] = proxy_basket_cad
    except Exception as e:
        print(f"Error calculating proxy: {e}")
        
    return metrics

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print("Fetching financial data from yfinance...")
    prices = fetch_latest_prices()
    
    print("Calculating metrics...")
    metrics = calculate_metrics(prices)
    
    output_path = os.path.join(DATA_DIR, "latest.json")
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"Successfully generated proxy nowcast data to {output_path}")

if __name__ == "__main__":
    main()
