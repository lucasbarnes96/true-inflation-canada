# TrueNorth Index V1
## One-Day Build PRD — Hobby Open Source Project

**Goal:** Build a live Canadian inflation tracker that combines a monthly anchor with motion-eligible public signals on a simple dashboard. Free forever. Open source. Deploy and ship fast.

---

## The Vision (V1 Scope)

A single-page dashboard that shows:
- **Current daily inflation estimate** (vs last month/year)
- **Category breakdown** (Food, Housing, Transport, Energy)
- **Live data sources** actively scraping
- **Comparison to official Statistics Canada CPI**

Nothing more. No auth, no payments, no complex infra. Just data → processing → display.

---

## Why This Can Compete with Truflation (Eventually)

Truflation's moat isn't their data sources (many are public). It's their **aggregation methodology** and **on-chain delivery**. 

Your advantages as a solo AI-powered builder:
1. **No corporate overhead** — move faster, pivot instantly
2. **AI-powered data extraction** — use LLMs to parse unstructured price data
3. **Community-driven** — open source means contributors can add data sources
4. **Novel data sources** — scrape where others buy APIs
5. **Transparent methodology** — everything is auditable on GitHub

---

## V1 Architecture (Dead Simple)

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Data Scrapers  │────▶│  Python Script  │────▶│  Static JSON    │
│  (4 sources)    │     │  (process.py)   │     │  (data.json)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │  HTML Dashboard │
                                               │  (GitHub Pages) │
                                               └─────────────────┘
```

**No database. No backend. No servers.** Just GitHub Actions running on a schedule.

---

## Platform: GitHub + GitHub Pages (100% Free)

| Component | Choice | Why |
|-----------|--------|-----|
| **Hosting** | GitHub Pages | Free, fast CDN, custom domain support |
| **Compute** | GitHub Actions | 2,000 minutes/month free (plenty) |
| **Data Storage** | GitHub repo (JSON files) | Versioned, free, accessible |
| **Scraping** | Python + requests/BeautifulSoup | No API keys needed |
| **Dashboard** | Vanilla HTML/CSS/JS + Chart.js | No build step, instant deploy |

---

## Data Sources (V1 — All Scrapable, All Free)

### 1. **Food Prices — OpenFoodFacts (API = Free)**
```python
# https://prices.openfoodfacts.org/api/v1/prices
# Returns: product name, price, location, date
# Coverage: 121,000+ prices, growing daily
```
- **Why:** Crowdsourced, open data, Canadian products included
- **Volume:** ~500 new prices/day
- **Weight in CPI:** 16.5%

### 2. **Gas Prices — CAA/NRCan (Scrape = Free)**
```python
# https://www.caa.ca/gas-prices/
# Or provincial sites like Ontario Energy Board
```
- **Why:** Updated daily, no API needed
- **Volume:** National + provincial averages
- **Weight in CPI:** ~4% (Transport component)

### 3. **Housing — CMHC + Scraping (Mixed)**
```python
# CMHC Open Data Portal (CSV downloads, free)
# https://www.cmhc-schl.gc.ca/en/professionals/housing-markets-data-and-research/housing-data
# + Scraping RentSeeker.ca or similar for current listings
```
- **Why:** CMHC is authoritative; scraping adds real-time signal
- **Weight in CPI:** ~30% (largest category)

### 4. **Electricity — Provincial Utilities (Scrape = Free)**
```python
# Ontario: https://www.oeb.ca/consumer-information/electricity-rates
# BC: https://www.bchydro.com/accounts-billing/rates-energy-use/electricity-rates.html
# Quebec: https://www.hydroquebec.com/residential/customer-space/rates/
```
- **Why:** Rates are public, change quarterly
- **Weight in CPI:** ~3% (Utilities component)

### Bonus: **Statistics Canada CPI (API = Free)**
```python
# pip install stats_can
# Official monthly CPI for validation/benchmarking
```
- **Use:** Compare your daily estimate to official monthly CPI
- **Why:** Ground truth for accuracy measurement

---

## The "Novel AI Techniques" (V1 Edition)

### 1. **LLM-Powered Price Extraction**
Instead of writing fragile regex for each site, use a small LLM (local or free tier) to extract prices:

```python
# Use Ollama locally or OpenAI API (free tier: $5-18 credit)
# Extract prices from messy HTML

prompt = """
Extract all product prices from this HTML snippet.
Return JSON: [{"product": "name", "price": 9.99, "unit": "per kg", "date": "2025-01-15"}]

HTML: {html_snippet}
"""
```

**Why novel:** Traditional scrapers break when sites change. LLMs are resilient to layout changes.

### 2. **Intelligent Category Mapping**
Use embeddings to automatically categorize products:

```python
# Product: "Organic Gala Apples, 3lb bag"
# Embedding → similarity search → "Food: Fresh Fruit"
# No manual rules needed
```

### 3. **Outlier Detection with AI**
```python
# Flag suspicious price changes for review
# "Milk went from $4.50 to $45.00" → likely data error
```

### 4. **Sentiment/News Scraping (Future)**
Scrape Canadian news for inflation-related headlines, use LLM to extract sentiment as a leading indicator.

---

## One-Day Build Plan

### Hour 1: Setup (15 min)
```bash
# Create GitHub repo
git init truenorth-index
cd truenorth-index

# Basic structure
mkdir -p scrapers data docs
touch scrapers/__init__.py

# Files needed:
# - scrapers/food.py (OpenFoodFacts)
# - scrapers/gas.py (CAA/NRCan)
# - scrapers/housing.py (CMHC + rent scraping)
# - scrapers/energy.py (Provincial utilities)
# - process.py (aggregation engine)
# - index.html (dashboard)
# - .github/workflows/scrape.yml (automation)
```

### Hour 2-3: Build Scrapers (2 hours)

**scraper_template.py:**
```python
import requests
import json
from datetime import datetime

def scrape_food_prices():
    """Scrape food prices from OpenFoodFacts"""
    url = "https://prices.openfoodfacts.org/api/v1/prices"
    params = {"page": 1, "size": 100, "location_country": "CA"}
    
    response = requests.get(url, params=params)
    data = response.json()
    
    prices = []
    for item in data.get("items", []):
        prices.append({
            "product": item.get("product_name", "Unknown"),
            "price": item.get("price"),
            "category": categorize_product(item.get("product_name")),
            "date": item.get("date"),
            "source": "openfoodfacts"
        })
    
    return prices

def categorize_product(name):
    """Use simple keyword matching (upgrade to embeddings later)"""
    name_lower = name.lower()
    if any(word in name_lower for word in ["milk", "cheese", "yogurt"]):
        return "food_dairy"
    elif any(word in name_lower for word in ["apple", "banana", "vegetable"]):
        return "food_produce"
    # ... more categories
    return "food_other"
```

### Hour 4: Build Aggregation Engine (1 hour)

**process.py:**
```python
import json
from datetime import datetime
from scrapers.food import scrape_food_prices
from scrapers.gas import scrape_gas_prices
from scrapers.housing import scrape_housing_data
from scrapers.energy import scrape_energy_prices

# CPI weights (approximate for V1)
WEIGHTS = {
    "food": 0.165,
    "housing": 0.300,
    "transport": 0.150,
    "energy": 0.080,
    # ... other categories
}

def calculate_daily_index():
    """Calculate daily inflation index from all sources"""
    
    # Collect data
    food_data = scrape_food_prices()
    gas_data = scrape_gas_prices()
    housing_data = scrape_housing_data()
    energy_data = scrape_energy_prices()
    
    # Calculate category averages
    food_avg = sum(p["price"] for p in food_data) / len(food_data) if food_data else 0
    gas_avg = sum(p["price"] for p in gas_data) / len(gas_data) if gas_data else 0
    
    # Load historical data for comparison
    try:
        with open("data/historical.json", "r") as f:
            historical = json.load(f)
    except FileNotFoundError:
        historical = {}
    
    # Calculate index (base = 100 on Jan 1, 2024)
    today = datetime.now().strftime("%Y-%m-%d")
    
    index_data = {
        "date": today,
        "timestamp": datetime.now().isoformat(),
        "categories": {
            "food": {"avg_price": food_avg, "data_points": len(food_data)},
            "gas": {"avg_price": gas_avg, "data_points": len(gas_data)},
            "housing": housing_data,
            "energy": energy_data
        },
        "total_data_points": len(food_data) + len(gas_data) + len(housing_data) + len(energy_data)
    }
    
    # Save to data.json (for dashboard)
    with open("data/latest.json", "w") as f:
        json.dump(index_data, f, indent=2)
    
    # Append to historical
    historical[today] = index_data
    with open("data/historical.json", "w") as f:
        json.dump(historical, f, indent=2)
    
    print(f"✓ Index calculated: {today}")
    print(f"✓ Total data points: {index_data['total_data_points']}")

if __name__ == "__main__":
    calculate_daily_index()
```

### Hour 5-6: Build Dashboard (2 hours)

**index.html:**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TrueNorth Index — Canadian Inflation Tracker</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff;
            min-height: 100vh;
            padding: 2rem;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        header { text-align: center; margin-bottom: 3rem; }
        h1 { font-size: 3rem; margin-bottom: 0.5rem; }
        h1 span { color: #00d4aa; }
        .subtitle { color: #8892b0; font-size: 1.1rem; }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-bottom: 3rem;
        }
        .stat-card {
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .stat-label { color: #8892b0; font-size: 0.875rem; text-transform: uppercase; }
        .stat-value { font-size: 2.5rem; font-weight: bold; margin: 0.5rem 0; }
        .stat-change { font-size: 1rem; }
        .positive { color: #ff6b6b; }
        .negative { color: #00d4aa; }
        .chart-container {
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }
        .data-sources {
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 1.5rem;
        }
        .source-item {
            display: flex;
            justify-content: space-between;
            padding: 0.75rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .source-status { color: #00d4aa; }
        footer {
            text-align: center;
            margin-top: 3rem;
            padding-top: 2rem;
            border-top: 1px solid rgba(255,255,255,0.1);
            color: #8892b0;
        }
        a { color: #00d4aa; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>TrueNorth <span>Index</span></h1>
            <p class="subtitle">Real-time Canadian inflation tracking • Updated daily</p>
        </header>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Daily Inflation Estimate</div>
                <div class="stat-value" id="current-rate">--%</div>
                <div class="stat-change" id="rate-change">vs last month</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Data Points Today</div>
                <div class="stat-value" id="data-points">--</div>
                <div class="stat-change">from 4 sources</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Official CPI (StatCan)</div>
                <div class="stat-value" id="official-cpi">--%</div>
                <div class="stat-change">last monthly release</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Last Updated</div>
                <div class="stat-value" id="last-update">--</div>
                <div class="stat-change">UTC</div>
            </div>
        </div>

        <div class="chart-container">
            <canvas id="inflationChart"></canvas>
        </div>

        <div class="data-sources">
            <h3>Active Data Sources</h3>
            <div class="source-item">
                <span>🍎 OpenFoodFacts (Food Prices)</span>
                <span class="source-status">● Live</span>
            </div>
            <div class="source-item">
                <span>⛽ CAA/NRCan (Gas Prices)</span>
                <span class="source-status">● Live</span>
            </div>
            <div class="source-item">
                <span>🏠 CMHC (Housing Data)</span>
                <span class="source-status">● Live</span>
            </div>
            <div class="source-item">
                <span>⚡ Provincial Utilities (Energy)</span>
                <span class="source-status">● Live</span>
            </div>
        </div>

        <footer>
            <p>Open source • <a href="https://github.com/YOUR_USERNAME/truenorth-index">GitHub</a></p>
            <p>Not financial advice • Data for educational purposes</p>
        </footer>
    </div>

    <script>
        // Fetch latest data
        fetch('data/latest.json')
            .then(r => r.json())
            .then(data => {
                document.getElementById('data-points').textContent = data.total_data_points.toLocaleString();
                document.getElementById('last-update').textContent = new Date(data.timestamp).toLocaleDateString();
            });

        // Fetch historical for chart
        fetch('data/historical.json')
            .then(r => r.json())
            .then(data => {
                const dates = Object.keys(data).slice(-30);
                const values = dates.map(d => data[d].categories.food.avg_price);
                
                new Chart(document.getElementById('inflationChart'), {
                    type: 'line',
                    data: {
                        labels: dates,
                        datasets: [{
                            label: 'Food Price Index',
                            data: values,
                            borderColor: '#00d4aa',
                            backgroundColor: 'rgba(0, 212, 170, 0.1)',
                            fill: true,
                            tension: 0.4
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: { legend: { labels: { color: '#fff' } } },
                        scales: {
                            x: { ticks: { color: '#8892b0' }, grid: { color: 'rgba(255,255,255,0.1)' } },
                            y: { ticks: { color: '#8892b0' }, grid: { color: 'rgba(255,255,255,0.1)' } }
                        }
                    }
                });
            });
    </script>
</body>
</html>
```

### Hour 7: Setup GitHub Actions (1 hour)

**.github/workflows/scrape.yml:**
```yaml
name: Daily Data Scrape

on:
  schedule:
    - cron: '0 6 * * *'  # Run daily at 6 AM UTC
  workflow_dispatch:  # Allow manual trigger

jobs:
  scrape:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install requests beautifulsoup4 pandas stats-can
    
    - name: Run scrapers
      run: python process.py
    
    - name: Commit and push
      run: |
        git config user.name "GitHub Action"
        git config user.email "action@github.com"
        git add data/
        git diff --quiet && git diff --staged --quiet || git commit -m "Update data: $(date +%Y-%m-%d)"
        git push
```

### Hour 8: Deploy & Test (1 hour)

1. **Enable GitHub Pages:**
   - Settings → Pages → Source: Deploy from branch → main → / (root)

2. **Test locally:**
   ```bash
   python process.py  # Generate data
   python -m http.server 8000  # Test dashboard
   ```

3. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Initial commit: V1 TrueNorth Index"
   git push origin main
   ```

4. **Verify:**
   - Check Actions tab for successful scrape
   - Visit `https://YOUR_USERNAME.github.io/truenorth-index`

---

## Post-Launch: Closing the Gap to Truflation

### Week 2-4: Accuracy Improvements
- [ ] Add more food categories (meat, dairy, grains)
- [ ] Implement proper CPI weighting
- [ ] Compare daily vs official CPI, calculate error rate
- [ ] Add trend smoothing (7-day moving average)

### Month 2: Scale Data Sources
- [ ] Scrape grocery store flyers (Loblaws, Sobeys, Metro)
- [ ] Add used car prices (AutoTrader scraping)
- [ ] Add airfare data (Google Flights scraping)
- [ ] Partner with price-tracking communities

### Month 3: AI-Powered Features
- [ ] LLM extraction for unstructured price data
- [ ] Embedding-based product categorization
- [ ] Predictive model (predict next month's CPI)
- [ ] Anomaly detection for data quality

### Month 6+: Advanced Features
- [ ] Provincial breakdowns
- [ ] City-level indexes (Toronto, Vancouver, Montreal)
- [ ] Category-specific indexes ("Egg Index", "Gas Index")
- [ ] API for developers
- [ ] Chainlink oracle integration

---

## Measuring Success vs Truflation

| Metric | Truflation | TrueNorth V1 Target |
|--------|-----------|---------------------|
| Update frequency | Daily | Daily |
| Data points | 18M+ | 1,000+ (V1) → 100K+ (Month 6) |
| Lead time | 45 days ahead of CPI | 30 days ahead (V1) |
| Categories | 12 | 4 (V1) → 12 (Month 6) |
| Cost to run | $$$$ | $0 |
| Open source | No | Yes |

**Your competitive advantage:** Transparency + community + AI-powered data extraction.

---

## Optional: Add AI Agents for Data Collection

Use n8n (free self-hosted) or Make.com to create agentic workflows:

```
Trigger (daily) → Scrape news → LLM extract price mentions → 
Validate against known sources → Add to database → Notify on anomalies
```

This is where AI agents shine: autonomously discovering new price signals.

---

## Summary: What You Ship in One Day

✅ **Live dashboard** showing daily Canadian inflation estimate  
✅ **4 data sources** actively scraping (food, gas, housing, energy)  
✅ **Automated updates** via GitHub Actions  
✅ **Free forever** hosting on GitHub Pages  
✅ **Open source** repo for community contributions  
✅ **Comparison** to official Statistics Canada CPI  

**Next:** Share on Reddit (r/PersonalFinanceCanada, r/fintech), Hacker News, Twitter. Build in public. Attract contributors. Iterate.

---

## Questions to Refine V1

1. **Should V1 include a simple prediction model?** (e.g., "Next month's CPI will be X%")
2. **Do you want provincial breakdowns in V1 or later?**
3. **Any specific categories you care most about?** (prioritize those)
4. **Want a "vs USA" comparison feature?** (compare Canadian inflation to US)
5. **Should we add a simple API endpoint in V1?** (JSON endpoint for other devs)

Let me know and I'll refine the PRD further!

---

## V1 Methodology Spec (Refined)

### Headline Definition
- The homepage headline is **"Daily CPI Nowcast (Estimate)"**, not official CPI, and it should be explained as an anchored headline plus live pulse.
- It represents a weighted nowcast of category proxy movement versus a prior-period baseline.
- It is published even when some categories are unavailable, with confidence downgraded and missing sources shown.

### Baseline and Formula
- Baseline period: previous available period in historical snapshots.
- Each category computes a proxy level from cleaned source observations.
- Category change: `((today_proxy / baseline_proxy) - 1) * 100`.
- Headline nowcast is an anchored level plus a weighted live pulse from motion-eligible category changes.
- Coverage ratio is weighted coverage of categories with usable data (`fresh` or `stale`).

### Confidence Rubric
- `high`: coverage ratio >= 0.90 with no anomaly flags.
- `medium`: coverage ratio >= 0.60 and < 0.90, or downgraded from `high` by anomaly flags.
- `low`: coverage ratio < 0.60, or downgraded from `medium` by anomaly flags.

### Data Quality Gates
- Deduplicate records by `source + item + date`.
- Enforce category-specific value bounds and reject invalid observations.
- Apply day-over-day outlier filtering using category median shift thresholds.
- Keep run metadata for accepted, rejected, and anomaly-filtered points.

### Source Hierarchy (Official/API First)
- Food: OpenFoodFacts API (tier 1).
- Transport: CAA gas page scrape fallback (tier 2).
- Housing: Statistics Canada CPI table (tier 1, monthly anchor proxy) plus rent pulse overlay when available.
- Energy: Ontario Energy Board rates scrape (tier 2).
- Official benchmark: Statistics Canada CPI table for monthly MoM and YoY comparison.

### V1 Scope Boundaries
- National Canada only.
- No provincial breakdown in V1.
- No prediction model/card in V1.
- No external API endpoint in V1 beyond repository JSON files.
- LLM extraction and embedding categorization are moved to post-V1 experimentation.
