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

try:
    from curl_cffi import requests as cffi_requests
    _CURL_AVAILABLE = True
except ImportError:
    _CURL_AVAILABLE = False

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


def fy_label(dt: Any) -> str:
    if hasattr(dt, "year"):
        return f"FY{str(dt.year)[2:]}"
    return str(dt)


# ---------------------------------------------------------------------------
# Yahoo Finance direct API (bypasses yfinance, uses curl_cffi Chrome spoof)
# ---------------------------------------------------------------------------
_YF_SESSION_OBJ: Any = None
_YF_CRUMB: str = ""
_YF_SESSION_TS: float = 0
_YF_HOST = "https://query1.finance.yahoo.com"


def _init_yahoo():
    """Get crumb + cookies from Yahoo Finance using Chrome TLS impersonation."""
    global _YF_SESSION_OBJ, _YF_CRUMB, _YF_SESSION_TS
    if not _CURL_AVAILABLE:
        print("[ROBU] curl_cffi unavailable — Yahoo Finance direct API disabled")
        return
    try:
        sess = cffi_requests.Session(impersonate="chrome120")
        sess.get("https://finance.yahoo.com", timeout=15)
        time.sleep(0.5)
        r = sess.get(f"{_YF_HOST}/v1/test/getcrumb", timeout=10)
        if r.status_code == 200 and r.text.strip():
            _YF_SESSION_OBJ = sess
            _YF_CRUMB = r.text.strip()
            _YF_SESSION_TS = time.time()
            print(f"[ROBU] Yahoo Finance session ready (crumb obtained)")
        else:
            print(f"[ROBU] Crumb fetch returned {r.status_code}: {r.text[:80]}")
    except Exception as e:
        print(f"[ROBU] Yahoo init error: {e}")


def _ensure_yahoo():
    """Re-init session if stale (>25 min)."""
    if not _YF_SESSION_OBJ or not _YF_CRUMB or time.time() - _YF_SESSION_TS > 1500:
        _init_yahoo()


def _yf_summary(symbol_ns: str, modules: str) -> dict:
    """Call Yahoo Finance quoteSummary directly."""
    _ensure_yahoo()
    if not _YF_SESSION_OBJ or not _YF_CRUMB:
        raise HTTPException(503, "Yahoo Finance session unavailable on this server")

    params = {"modules": modules, "crumb": _YF_CRUMB, "formatted": "false", "lang": "en-US"}
    resp = _YF_SESSION_OBJ.get(
        f"{_YF_HOST}/v10/finance/quoteSummary/{symbol_ns}", params=params, timeout=20
    )

    # If session expired, refresh once and retry
    if resp.status_code in (401, 403):
        global _YF_SESSION_TS
        _YF_SESSION_TS = 0
        _init_yahoo()
        if not _YF_SESSION_OBJ or not _YF_CRUMB:
            raise HTTPException(503, "Yahoo Finance session refresh failed")
        params["crumb"] = _YF_CRUMB
        resp = _YF_SESSION_OBJ.get(
            f"{_YF_HOST}/v10/finance/quoteSummary/{symbol_ns}", params=params, timeout=20
        )

    if not resp.ok:
        raise HTTPException(resp.status_code, f"Yahoo Finance error for {symbol_ns}: HTTP {resp.status_code}")

    data = resp.json()
    err = (data.get("quoteSummary") or {}).get("error")
    if err:
        raise HTTPException(404, err.get("description", "Not found"))
    results = (data.get("quoteSummary") or {}).get("result") or []
    if not results:
        raise HTTPException(404, f"No data for {symbol_ns}")
    return results[0]


# Initialise Yahoo session at startup
_init_yahoo()


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
    """Return key company metrics — direct Yahoo Finance API via curl_cffi."""
    symbol = symbol.upper().strip()
    cache_key = f"company:{symbol}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    ns = symbol + ".NS"
    modules = "financialData,defaultKeyStatistics,assetProfile,summaryDetail,price"
    data = _yf_summary(ns, modules)

    fd  = data.get("financialData", {})
    ks  = data.get("defaultKeyStatistics", {})
    ap  = data.get("assetProfile", {})
    sd  = data.get("summaryDetail", {})
    pr  = data.get("price", {})

    def rv(d: dict, key: str) -> Any:
        v = d.get(key)
        if isinstance(v, dict):
            return v.get("raw")
        return v

    price_val = safe_val(rv(pr, "regularMarketPrice") or rv(fd, "currentPrice"))
    if price_val == 0:
        raise HTTPException(404, f"No price data for {ns}")

    prev_close = safe_val(rv(pr, "regularMarketPreviousClose") or rv(sd, "previousClose"))
    change = round(price_val - prev_close, 2) if prev_close else 0.0
    change_pct = round((change / prev_close) * 100, 2) if prev_close else 0.0
    mktcap_cr = round(safe_val(rv(pr, "marketCap") or rv(sd, "marketCap")) / 1e7, 2)
    shares_cr = round(safe_val(rv(ks, "sharesOutstanding")) / 1e7, 2)

    stock_meta = STOCK_UNIVERSE.get(symbol, {})
    roe_raw = rv(fd, "returnOnEquity")
    roa_raw = rv(fd, "returnOnAssets")
    div_raw = rv(sd, "dividendYield")
    rev_g   = rv(fd, "revenueGrowth")
    ear_g   = rv(fd, "earningsGrowth")
    d2e     = rv(fd, "debtToEquity")

    result = {
        "symbol": symbol,
        "name": rv(pr, "longName") or rv(pr, "shortName") or stock_meta.get("name", symbol),
        "sector": ap.get("sector") or stock_meta.get("sector", "Unknown"),
        "industry": ap.get("industry", ""),
        "currentPrice": price_val,
        "previousClose": prev_close,
        "change": change,
        "changePct": change_pct,
        "marketCap": mktcap_cr,
        "pe": safe_val(rv(sd, "trailingPE")),
        "forwardPE": safe_val(rv(sd, "forwardPE")),
        "pb": safe_val(rv(ks, "priceToBook")),
        "roe": round(float(roe_raw) * 100, 2) if roe_raw else 0.0,
        "roa": round(float(roa_raw) * 100, 2) if roa_raw else 0.0,
        "eps": safe_val(rv(ks, "trailingEps")),
        "dividendYield": round(float(div_raw) * 100, 2) if div_raw else 0.0,
        "week52High": safe_val(rv(sd, "fiftyTwoWeekHigh")),
        "week52Low": safe_val(rv(sd, "fiftyTwoWeekLow")),
        "debtToEquity": round(safe_val(d2e) / 100, 2) if d2e else 0.0,
        "currentRatio": safe_val(rv(fd, "currentRatio")),
        "shares": shares_cr,
        "beta": safe_val(rv(ks, "beta")),
        "revenueGrowth": round(float(rev_g) * 100, 2) if rev_g else 0.0,
        "earningsGrowth": round(float(ear_g) * 100, 2) if ear_g else 0.0,
    }

    _cache_set(cache_key, result)
    return result


@app.get("/financials/{symbol}")
def financials(symbol: str):
    """Return last 5 years of annual financials — direct Yahoo Finance API."""
    symbol = symbol.upper().strip()
    cache_key = f"financials:{symbol}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    ns = symbol + ".NS"
    data = _yf_summary(ns, "incomeStatementHistory,defaultKeyStatistics")
    stmts = (data.get("incomeStatementHistory") or {}).get("incomeStatementHistory") or []

    if not stmts:
        raise HTTPException(404, f"No financials for {ns}")

    stmts = sorted(stmts, key=lambda x: x.get("endDate", {}).get("raw", 0))[-5:]

    # Shares outstanding — used to calculate EPS when Yahoo doesn't provide it directly
    ks = data.get("defaultKeyStatistics") or {}
    so_raw = ks.get("sharesOutstanding")
    shares_outstanding = float(so_raw.get("raw", 0) if isinstance(so_raw, dict) else (so_raw or 0))
    # trailingEps from key stats — most reliable EPS figure for the latest year
    teps_raw = ks.get("trailingEps")
    trailing_eps = float(teps_raw.get("raw", 0) if isinstance(teps_raw, dict) else (teps_raw or 0))

    def rv(d: dict, key: str) -> float:
        v = d.get(key)
        if isinstance(v, dict):
            return safe_val(v.get("raw"))
        return safe_val(v)

    rows = []
    prev_revenue = None

    for i, stmt in enumerate(stmts):
        end_ts = (stmt.get("endDate") or {}).get("raw", 0)
        year_label = f"FY{datetime.utcfromtimestamp(end_ts).strftime('%y')}" if end_ts else "?"

        # totalRevenue is 0 for banks — fall back to interest income
        revenue = (rv(stmt, "totalRevenue")
                   or rv(stmt, "totalInterestIncome")
                   or rv(stmt, "netInterestIncome")
                   or rv(stmt, "totalIncome"))
        pat     = rv(stmt, "netIncome")
        ebitda  = rv(stmt, "ebitda")
        gross   = rv(stmt, "grossProfit")
        op_inc  = rv(stmt, "operatingIncome")
        da      = rv(stmt, "depreciationAndAmortization")

        # Try direct EPS fields (Yahoo Finance is inconsistent with casing)
        eps_val = (rv(stmt, "basicEPS") or rv(stmt, "dilutedEPS")
                   or rv(stmt, "basicEps") or rv(stmt, "dilutedEps"))

        # Fallback: calculate EPS from PAT ÷ shares outstanding
        # PAT from stmt is in absolute rupees; shares_outstanding is also absolute
        if not eps_val and pat and shares_outstanding > 0:
            eps_val = pat / shares_outstanding

        # For the most recent year, prefer trailingEps which is most accurate
        if i == len(stmts) - 1 and trailing_eps > 0:
            eps_val = trailing_eps

        if ebitda == 0 and op_inc and da:
            ebitda = op_inc + da
        elif ebitda == 0 and gross:
            ebitda = gross

        revenue_cr   = round(revenue / 1e7, 2) if revenue else 0.0
        pat_cr       = round(pat / 1e7, 2) if pat else 0.0
        ebitda_cr    = round(ebitda / 1e7, 2) if ebitda else 0.0
        net_margin   = round((pat / revenue) * 100, 2) if revenue else 0.0
        ebitda_margin= round((ebitda / revenue) * 100, 2) if revenue and ebitda else 0.0
        rev_growth   = 0.0
        if prev_revenue and prev_revenue > 0 and revenue:
            rev_growth = round(((revenue - prev_revenue) / prev_revenue) * 100, 2)
        prev_revenue = revenue

        rows.append({
            "year": year_label,
            "revenue": revenue_cr,
            "pat": pat_cr,
            "ebitda": ebitda_cr,
            "eps": round(eps_val, 2) if eps_val else 0.0,
            "netMargin": net_margin,
            "revenueGrowth": rev_growth,
            "ebitdaMargin": ebitda_margin,
        })

    _cache_set(cache_key, rows)
    return rows


@app.get("/price/{symbol}")
def price(symbol: str):
    """Quick current price endpoint — direct Yahoo Finance API."""
    symbol = symbol.upper().strip()
    cache_key = f"price:{symbol}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    ns = symbol + ".NS"
    data = _yf_summary(ns, "price")
    pr = data.get("price", {})

    def rv(d: dict, key: str) -> Any:
        v = d.get(key)
        return v.get("raw") if isinstance(v, dict) else v

    current_price = safe_val(rv(pr, "regularMarketPrice"))
    prev_close = safe_val(rv(pr, "regularMarketPreviousClose"))
    change = round(current_price - prev_close, 2) if current_price and prev_close else 0.0
    change_pct = round((change / prev_close) * 100, 2) if prev_close else 0.0

    result = {"symbol": symbol, "price": current_price, "change": change, "changePct": change_pct}
    _cache_set(cache_key, result)
    return result


@app.get("/universe/size")
def universe_size():
    return {"count": len(STOCK_UNIVERSE), "source": "NSE EQUITY_L.csv"}
