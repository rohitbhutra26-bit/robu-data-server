"""
ROBU Data Server
FastAPI server providing Indian stock data via yfinance.
Supports full NSE universe (~2000 stocks) downloaded from NSE on startup.

Run: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
import time
import os
import json
import requests
from datetime import datetime, timedelta
from typing import Any

app = FastAPI(title="ROBU Data Server", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# NSE Full Universe — downloaded from NSE on startup, cached locally 24h
# ---------------------------------------------------------------------------
_NSE_CSV_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
_NSE_CACHE_FILE = os.path.join(os.path.dirname(__file__), "nse_equity_list.json")
_NSE_CACHE_TTL_HOURS = 24

# Fallback list if NSE CSV download fails
_FALLBACK_STOCKS: dict[str, dict] = {
    "RELIANCE":   {"name": "Reliance Industries Ltd",           "sector": "Energy"},
    "TCS":        {"name": "Tata Consultancy Services Ltd",     "sector": "Information Technology"},
    "INFY":       {"name": "Infosys Ltd",                       "sector": "Information Technology"},
    "HDFCBANK":   {"name": "HDFC Bank Ltd",                     "sector": "Banking"},
    "ICICIBANK":  {"name": "ICICI Bank Ltd",                    "sector": "Banking"},
    "WIPRO":      {"name": "Wipro Ltd",                         "sector": "Information Technology"},
    "BAJFINANCE": {"name": "Bajaj Finance Ltd",                 "sector": "NBFC"},
    "HINDUNILVR": {"name": "Hindustan Unilever Ltd",            "sector": "FMCG"},
    "ITC":        {"name": "ITC Ltd",                           "sector": "FMCG"},
    "KOTAKBANK":  {"name": "Kotak Mahindra Bank Ltd",           "sector": "Banking"},
    "LT":         {"name": "Larsen & Toubro Ltd",               "sector": "Infrastructure"},
    "AXISBANK":   {"name": "Axis Bank Ltd",                     "sector": "Banking"},
    "ASIANPAINT": {"name": "Asian Paints Ltd",                  "sector": "Consumer"},
    "MARUTI":     {"name": "Maruti Suzuki India Ltd",           "sector": "Automobiles"},
    "TITAN":      {"name": "Titan Company Ltd",                 "sector": "Consumer"},
    "SUNPHARMA":  {"name": "Sun Pharmaceutical Industries Ltd", "sector": "Pharma"},
    "HCLTECH":    {"name": "HCL Technologies Ltd",              "sector": "Information Technology"},
    "TATAMOTORS": {"name": "Tata Motors Ltd",                   "sector": "Automobiles"},
    "TATASTEEL":  {"name": "Tata Steel Ltd",                    "sector": "Metals"},
    "SBIN":       {"name": "State Bank of India",               "sector": "Banking"},
    "ADANIENT":   {"name": "Adani Enterprises Ltd",             "sector": "Conglomerate"},
    "BHARTIARTL": {"name": "Bharti Airtel Ltd",                 "sector": "Telecom"},
    "KAYNES":     {"name": "Kaynes Technology India Ltd",       "sector": "Electronics"},
    "NTPC":       {"name": "NTPC Ltd",                          "sector": "Utilities"},
    "ONGC":       {"name": "Oil & Natural Gas Corporation Ltd", "sector": "Energy"},
}

# In-memory stock universe: {SYMBOL: {name, sector}}
STOCK_UNIVERSE: dict[str, dict] = {}


def _load_nse_universe():
    """Download full NSE equity list and cache it. Falls back to hardcoded list."""
    global STOCK_UNIVERSE

    # Check local cache first
    if os.path.exists(_NSE_CACHE_FILE):
        try:
            with open(_NSE_CACHE_FILE) as f:
                cached = json.load(f)
            saved_at = datetime.fromisoformat(cached.get("saved_at", "2000-01-01"))
            if datetime.now() - saved_at < timedelta(hours=_NSE_CACHE_TTL_HOURS):
                STOCK_UNIVERSE = cached["stocks"]
                print(f"[ROBU] Loaded {len(STOCK_UNIVERSE)} stocks from local cache.")
                return
        except Exception:
            pass

    # Try downloading from NSE
    try:
        print("[ROBU] Downloading NSE equity list...")
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "text/csv,*/*",
            "Referer": "https://www.nseindia.com",
        }
        resp = requests.get(_NSE_CSV_URL, headers=headers, timeout=30)
        resp.raise_for_status()

        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))

        # NSE CSV columns: SYMBOL, NAME OF COMPANY, SERIES, ...
        universe = {}
        for _, row in df.iterrows():
            sym = str(row.get("SYMBOL", "")).strip().upper()
            name = str(row.get("NAME OF COMPANY", "")).strip()
            if sym and name and len(sym) > 0:
                universe[sym] = {"name": name, "sector": "NSE Listed"}

        if len(universe) > 100:
            STOCK_UNIVERSE = universe
            # Save to local cache
            with open(_NSE_CACHE_FILE, "w") as f:
                json.dump({"saved_at": datetime.now().isoformat(), "stocks": universe}, f)
            print(f"[ROBU] Downloaded {len(universe)} NSE stocks.")
            return

    except Exception as e:
        print(f"[ROBU] NSE download failed: {e}. Using fallback list.")

    STOCK_UNIVERSE = _FALLBACK_STOCKS
    print(f"[ROBU] Using fallback list of {len(STOCK_UNIVERSE)} stocks.")


# Load universe on startup
_load_nse_universe()


# ---------------------------------------------------------------------------
# Simple TTL cache — {key: (data, timestamp)}
# ---------------------------------------------------------------------------
_cache: dict[str, tuple[Any, float]] = {}
CACHE_TTL = 900  # 15 minutes


def _cache_get(key: str) -> Any | None:
    entry = _cache.get(key)
    if entry is None:
        return None
    data, ts = entry
    if time.time() - ts > CACHE_TTL:
        del _cache[key]
        return None
    return data


def _cache_set(key: str, data: Any) -> None:
    _cache[key] = (data, time.time())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_val(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return default
        return round(f, 2)
    except (TypeError, ValueError):
        return default


def get_ticker(symbol: str) -> yf.Ticker:
    return yf.Ticker(symbol + ".NS")


def fy_label(dt: Any) -> str:
    if hasattr(dt, "year"):
        return f"FY{str(dt.year)[2:]}"
    return str(dt)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "universe_size": len(STOCK_UNIVERSE)}


@app.get("/search")
def search(q: str = ""):
    """Search full NSE universe by symbol or company name."""
    q = q.strip()
    if not q:
        # Return top 10 by default (popular stocks first)
        top = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
               "WIPRO", "BAJFINANCE", "ITC", "SBIN", "TATAMOTORS"]
        results = []
        for sym in top:
            if sym in STOCK_UNIVERSE:
                results.append({"symbol": sym, **STOCK_UNIVERSE[sym]})
        return results

    q_lower = q.lower()
    exact, starts, contains = [], [], []

    for sym, info in STOCK_UNIVERSE.items():
        sym_lower = sym.lower()
        name_lower = info["name"].lower()
        if sym_lower == q_lower:
            exact.append({"symbol": sym, **info})
        elif sym_lower.startswith(q_lower) or name_lower.startswith(q_lower):
            starts.append({"symbol": sym, **info})
        elif q_lower in sym_lower or q_lower in name_lower:
            contains.append({"symbol": sym, **info})

    # Best matches first
    results = (exact + starts + contains)[:20]
    return results


@app.get("/company/{symbol}")
def company(symbol: str):
    """Return key company metrics from yfinance for any NSE stock."""
    symbol = symbol.upper().strip()
    cache_key = f"company:{symbol}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        ticker = get_ticker(symbol)
        time.sleep(0.5)
        info = ticker.info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"yfinance error for {symbol}: {str(e)}")

    price = safe_val(info.get("currentPrice") or info.get("regularMarketPrice"))
    if price == 0:
        raise HTTPException(status_code=404, detail=f"No price data for {symbol}.NS — check symbol is correct.")

    prev_close = safe_val(info.get("previousClose") or info.get("regularMarketPreviousClose"))
    change = round(price - prev_close, 2) if prev_close else 0.0
    change_pct = round((change / prev_close) * 100, 2) if prev_close else 0.0

    mktcap_cr = round(safe_val(info.get("marketCap")) / 1e7, 2)
    shares_cr = round(safe_val(info.get("sharesOutstanding")) / 1e7, 2)

    stock_meta = STOCK_UNIVERSE.get(symbol, {})

    result = {
        "symbol": symbol,
        "name": info.get("longName") or info.get("shortName") or stock_meta.get("name", symbol),
        "sector": info.get("sector") or stock_meta.get("sector", "Unknown"),
        "industry": info.get("industry", ""),
        "currentPrice": price,
        "previousClose": prev_close,
        "change": change,
        "changePct": change_pct,
        "marketCap": mktcap_cr,
        "pe": safe_val(info.get("trailingPE")),
        "forwardPE": safe_val(info.get("forwardPE")),
        "pb": safe_val(info.get("priceToBook")),
        "roe": round(safe_val(info.get("returnOnEquity")) * 100, 2) if info.get("returnOnEquity") else 0.0,
        "roa": round(safe_val(info.get("returnOnAssets")) * 100, 2) if info.get("returnOnAssets") else 0.0,
        "eps": safe_val(info.get("trailingEps")),
        "dividendYield": round(safe_val(info.get("dividendYield")) * 100, 2) if info.get("dividendYield") else 0.0,
        "week52High": safe_val(info.get("fiftyTwoWeekHigh")),
        "week52Low": safe_val(info.get("fiftyTwoWeekLow")),
        # yfinance returns debtToEquity as percentage (e.g. 10.5 = 10.5% = 0.105x ratio)
        # Divide by 100 to get the actual ratio displayed to users
        "debtToEquity": round(safe_val(info.get("debtToEquity")) / 100, 2),
        "currentRatio": safe_val(info.get("currentRatio")),
        "shares": shares_cr,
        "beta": safe_val(info.get("beta")),
        "revenueGrowth": round(safe_val(info.get("revenueGrowth")) * 100, 2) if info.get("revenueGrowth") else 0.0,
        "earningsGrowth": round(safe_val(info.get("earningsGrowth")) * 100, 2) if info.get("earningsGrowth") else 0.0,
    }

    _cache_set(cache_key, result)
    return result


@app.get("/financials/{symbol}")
def financials(symbol: str):
    """Return last 5 years of annual financials for any NSE stock."""
    symbol = symbol.upper().strip()
    cache_key = f"financials:{symbol}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        ticker = get_ticker(symbol)
        time.sleep(0.5)
        fin = ticker.financials
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"yfinance financials error for {symbol}: {str(e)}")

    if fin is None or fin.empty:
        raise HTTPException(status_code=404, detail=f"No financials for {symbol}.NS")

    fin = fin[sorted(fin.columns)]
    cols = list(fin.columns)[-5:]
    fin = fin[cols]

    rows = []
    prev_revenue = None

    for col in cols:
        def g(key: str) -> float:
            if key in fin.index:
                return safe_val(fin.loc[key, col])
            return 0.0

        revenue = g("Total Revenue")
        pat = g("Net Income")
        ebitda = g("EBITDA")

        if ebitda == 0.0:
            op_income = g("Operating Income")
            da = g("Reconciled Depreciation") or g("Depreciation And Amortization In Income Statement")
            gross = g("Gross Profit")
            if op_income and da:
                ebitda = op_income + da
            elif gross:
                sga = g("Selling General And Administration")
                ebitda = gross - sga if sga else gross

        eps_val = g("Basic EPS") or g("Diluted EPS")

        revenue_cr = round(revenue / 1e7, 2) if revenue else 0.0
        pat_cr = round(pat / 1e7, 2) if pat else 0.0
        ebitda_cr = round(ebitda / 1e7, 2) if ebitda else 0.0

        net_margin = round((pat / revenue) * 100, 2) if revenue else 0.0
        ebitda_margin = round((ebitda / revenue) * 100, 2) if revenue and ebitda else 0.0

        rev_growth = 0.0
        if prev_revenue and prev_revenue > 0 and revenue:
            rev_growth = round(((revenue - prev_revenue) / prev_revenue) * 100, 2)
        prev_revenue = revenue

        rows.append({
            "year": fy_label(col),
            "revenue": revenue_cr,
            "pat": pat_cr,
            "ebitda": ebitda_cr,
            "eps": round(float(eps_val), 2) if eps_val else 0.0,
            "netMargin": net_margin,
            "revenueGrowth": rev_growth,
            "ebitdaMargin": ebitda_margin,
        })

    _cache_set(cache_key, rows)
    return rows


@app.get("/price/{symbol}")
def price(symbol: str):
    """Quick current price endpoint."""
    symbol = symbol.upper().strip()
    cache_key = f"price:{symbol}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        ticker = get_ticker(symbol)
        time.sleep(0.3)
        info = ticker.info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    current_price = safe_val(info.get("currentPrice") or info.get("regularMarketPrice"))
    prev_close = safe_val(info.get("previousClose") or info.get("regularMarketPreviousClose"))
    change = round(current_price - prev_close, 2) if current_price and prev_close else 0.0
    change_pct = round((change / prev_close) * 100, 2) if prev_close else 0.0

    result = {"symbol": symbol, "price": current_price, "change": change, "changePct": change_pct}
    _cache_set(cache_key, result)
    return result


@app.get("/universe/size")
def universe_size():
    return {"count": len(STOCK_UNIVERSE), "source": "NSE EQUITY_L.csv"}
