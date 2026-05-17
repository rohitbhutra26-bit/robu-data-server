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
_BSE_LIST_URL = "https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w?Group=&Scripcode=&industry=&segment=Equity&status=Active"
_NSE_CACHE_FILE = os.path.join(os.path.dirname(__file__), "nse_equity_list.json")
_NSE_CACHE_TTL_HOURS = 24

# Fallback list if NSE CSV download fails
def _nse(name: str, sector: str, sym: str) -> dict:
    return {"name": name, "sector": sector, "exchange": "NSE", "yf_ticker": f"{sym}.NS"}

_FALLBACK_STOCKS: dict[str, dict] = {
    "RELIANCE":   _nse("Reliance Industries Ltd",           "Energy",                   "RELIANCE"),
    "TCS":        _nse("Tata Consultancy Services Ltd",     "Information Technology",   "TCS"),
    "INFY":       _nse("Infosys Ltd",                       "Information Technology",   "INFY"),
    "HDFCBANK":   _nse("HDFC Bank Ltd",                     "Banking",                  "HDFCBANK"),
    "ICICIBANK":  _nse("ICICI Bank Ltd",                    "Banking",                  "ICICIBANK"),
    "WIPRO":      _nse("Wipro Ltd",                         "Information Technology",   "WIPRO"),
    "BAJFINANCE": _nse("Bajaj Finance Ltd",                 "NBFC",                     "BAJFINANCE"),
    "HINDUNILVR": _nse("Hindustan Unilever Ltd",            "FMCG",                     "HINDUNILVR"),
    "ITC":        _nse("ITC Ltd",                           "FMCG",                     "ITC"),
    "KOTAKBANK":  _nse("Kotak Mahindra Bank Ltd",           "Banking",                  "KOTAKBANK"),
    "LT":         _nse("Larsen & Toubro Ltd",               "Infrastructure",           "LT"),
    "AXISBANK":   _nse("Axis Bank Ltd",                     "Banking",                  "AXISBANK"),
    "ASIANPAINT": _nse("Asian Paints Ltd",                  "Consumer",                 "ASIANPAINT"),
    "MARUTI":     _nse("Maruti Suzuki India Ltd",           "Automobiles",              "MARUTI"),
    "TITAN":      _nse("Titan Company Ltd",                 "Consumer",                 "TITAN"),
    "SUNPHARMA":  _nse("Sun Pharmaceutical Industries Ltd", "Pharmaceuticals",          "SUNPHARMA"),
    "HCLTECH":    _nse("HCL Technologies Ltd",              "Information Technology",   "HCLTECH"),
    "TATAMOTORS": _nse("Tata Motors Ltd",                   "Automobiles",              "TATAMOTORS"),
    "TATASTEEL":  _nse("Tata Steel Ltd",                    "Metals",                   "TATASTEEL"),
    "SBIN":       _nse("State Bank of India",               "Banking",                  "SBIN"),
    "ADANIENT":   _nse("Adani Enterprises Ltd",             "Conglomerate",             "ADANIENT"),
    "BHARTIARTL": _nse("Bharti Airtel Ltd",                 "Telecom",                  "BHARTIARTL"),
    "KAYNES":     _nse("Kaynes Technology India Ltd",       "Electronics",              "KAYNES"),
    "NTPC":       _nse("NTPC Ltd",                          "Utilities",                "NTPC"),
    "ONGC":       _nse("Oil & Natural Gas Corporation Ltd", "Energy",                   "ONGC"),
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

        # NSE CSV columns: SYMBOL, NAME OF COMPANY, SERIES, ISIN NUMBER, ...
        universe = {}
        for _, row in df.iterrows():
            sym = str(row.get("SYMBOL", "")).strip().upper()
            name = str(row.get("NAME OF COMPANY", "")).strip()
            isin = str(row.get("ISIN NUMBER", "")).strip()
            if sym and name and len(sym) > 0:
                universe[sym] = {
                    "name": name,
                    "sector": "NSE Listed",
                    "exchange": "NSE",
                    "isin": isin,
                    "yf_ticker": f"{sym}.NS",
                }

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


def _load_bse_universe():
    """Fetch BSE equity list and add BSE-only stocks (not already in NSE) to universe."""
    global STOCK_UNIVERSE
    # Build ISIN → NSE symbol map from current universe
    isin_to_nse = {
        info.get("isin", ""): sym
        for sym, info in STOCK_UNIVERSE.items()
        if info.get("isin")
    }
    try:
        if _CURL_AVAILABLE:
            _bse_sess = cffi_requests.Session(impersonate="chrome120")
            _bse_sess.get("https://www.bseindia.com", timeout=15)
            time.sleep(0.5)
            resp = _bse_sess.get(_BSE_LIST_URL, headers={"Accept": "application/json"}, timeout=30)
        else:
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                "Referer": "https://www.bseindia.com/",
                "Accept": "application/json, */*",
            }
            resp = requests.get(_BSE_LIST_URL, headers=headers, timeout=30)
        if not resp.ok:
            print(f"[ROBU] BSE list HTTP {resp.status_code} — skipping BSE universe")
            return
        items = resp.json()
        if not isinstance(items, list):
            print("[ROBU] BSE list unexpected format — skipping")
            return

        added = 0
        for item in items:
            code = str(item.get("Scripcode", "")).strip()
            name = str(item.get("Scrip_Name", "")).strip()
            isin = str(item.get("ISIN_NO", "")).strip()
            sector = str(item.get("industry", "BSE Listed")).strip() or "BSE Listed"
            if not code or not name:
                continue
            # Already in NSE universe (matched by ISIN) — skip, NSE is primary
            if isin and isin in isin_to_nse:
                continue
            # BSE-only stock — add with numeric code as identifier
            if code not in STOCK_UNIVERSE:
                STOCK_UNIVERSE[code] = {
                    "name": name,
                    "sector": sector,
                    "exchange": "BSE",
                    "isin": isin,
                    "yf_ticker": f"{code}.BO",
                }
                added += 1
        print(f"[ROBU] BSE universe merged: {added} BSE-only stocks added")
    except Exception as e:
        print(f"[ROBU] BSE universe load failed: {e}")


def _get_yf_ticker(symbol: str) -> str:
    """Return the correct Yahoo Finance ticker for a symbol (e.g. TCS→TCS.NS, 543652→543652.BO)."""
    info = STOCK_UNIVERSE.get(symbol, {})
    if info.get("yf_ticker"):
        return info["yf_ticker"]
    # Numeric-only symbols are BSE script codes
    if symbol.isdigit():
        return f"{symbol}.BO"
    return f"{symbol}.NS"


# Load universe on startup
_load_nse_universe()
_load_bse_universe()


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
    """Get crumb + cookies from Yahoo Finance using Chrome TLS impersonation.
    Retries up to 5 times with backoff if rate-limited (429)."""
    global _YF_SESSION_OBJ, _YF_CRUMB, _YF_SESSION_TS
    if not _CURL_AVAILABLE:
        print("[ROBU] curl_cffi unavailable — Yahoo Finance direct API disabled")
        return
    delays = [2, 5, 15, 30, 60]  # seconds between retries
    for attempt, delay in enumerate(delays, 1):
        try:
            sess = cffi_requests.Session(impersonate="chrome120")
            sess.get("https://finance.yahoo.com", timeout=15)
            time.sleep(1)
            r = sess.get(f"{_YF_HOST}/v1/test/getcrumb", timeout=10)
            if r.status_code == 200 and r.text.strip():
                _YF_SESSION_OBJ = sess
                _YF_CRUMB = r.text.strip()
                _YF_SESSION_TS = time.time()
                print(f"[ROBU] Yahoo Finance session ready (attempt {attempt})")
                return
            elif r.status_code == 429:
                print(f"[ROBU] Yahoo 429 rate-limit — waiting {delay}s before retry {attempt}/{len(delays)}")
                time.sleep(delay)
            else:
                print(f"[ROBU] Crumb fetch returned {r.status_code}: {r.text[:80]}")
                time.sleep(delay)
        except Exception as e:
            print(f"[ROBU] Yahoo init error (attempt {attempt}): {e}")
            time.sleep(delay)
    print("[ROBU] Yahoo Finance session failed after all retries — data unavailable until next request")


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


def _search_row(sym: str, info: dict) -> dict:
    """Return clean search result — only fields the frontend needs."""
    return {
        "symbol": sym,
        "name": info.get("name", sym),
        "sector": info.get("sector", ""),
        "exchange": info.get("exchange", "NSE"),
    }


def _yahoo_search(q: str) -> list[dict]:
    """
    Hit Yahoo Finance's search API to find Indian stocks (NSE + BSE).
    Works even when BSE equity list API is blocked — Yahoo indexes everything.
    Results are cached into STOCK_UNIVERSE so subsequent /company calls work.
    """
    try:
        _ensure_yahoo()
        if not _YF_SESSION_OBJ:
            return []
        resp = _YF_SESSION_OBJ.get(
            "https://query2.finance.yahoo.com/v1/finance/search",
            params={
                "q": q,
                "lang": "en-US",
                "region": "IN",
                "quotesCount": 12,
                "newsCount": 0,
                "listsCount": 0,
                "crumb": _YF_CRUMB,
            },
            timeout=10,
        )
        if not resp.ok:
            return []
        quotes = resp.json().get("quotes", [])
        results = []
        for qt in quotes:
            raw_sym = qt.get("symbol", "")
            if not raw_sym:
                continue
            # Only Indian exchange tickers
            if not (raw_sym.endswith(".NS") or raw_sym.endswith(".BO")):
                continue
            exchange = "BSE" if raw_sym.endswith(".BO") else "NSE"
            clean_sym = raw_sym[:-3]  # strip .NS or .BO
            name = qt.get("shortname") or qt.get("longname") or clean_sym
            sector = qt.get("industry") or qt.get("sector") or f"{exchange} Listed"
            # Register in universe so /company lookup works
            if clean_sym not in STOCK_UNIVERSE:
                STOCK_UNIVERSE[clean_sym] = {
                    "name": name,
                    "sector": sector,
                    "exchange": exchange,
                    "yf_ticker": raw_sym,
                }
            results.append({
                "symbol": clean_sym,
                "name": name,
                "sector": sector,
                "exchange": exchange,
            })
        return results
    except Exception as e:
        print(f"[ROBU] Yahoo search fallback error: {e}")
        return []


@app.get("/search")
def search(q: str = ""):
    """Search NSE + BSE universe by symbol or company name.
    Falls back to Yahoo Finance search API for stocks not in local universe."""
    q = q.strip()
    if not q:
        top = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
               "WIPRO", "BAJFINANCE", "ITC", "SBIN", "TATAMOTORS"]
        return [_search_row(sym, STOCK_UNIVERSE[sym]) for sym in top if sym in STOCK_UNIVERSE]

    q_lower = q.lower()
    exact, starts, contains = [], [], []

    for sym, info in STOCK_UNIVERSE.items():
        sym_lower = sym.lower()
        name_lower = info.get("name", "").lower()
        row = _search_row(sym, info)
        if sym_lower == q_lower:
            exact.append(row)
        elif sym_lower.startswith(q_lower) or name_lower.startswith(q_lower):
            starts.append(row)
        elif q_lower in sym_lower or q_lower in name_lower:
            contains.append(row)

    # NSE results first within each tier, then BSE
    def _rank(r): return 0 if r["exchange"] == "NSE" else 1
    exact.sort(key=_rank)
    starts.sort(key=_rank)
    contains.sort(key=_rank)
    local = (exact + starts + contains)[:20]

    # If local universe has few hits, augment with Yahoo Finance live search
    # This catches BSE-only stocks (SME, exclusive listings) not in local cache
    if len(local) < 6 and len(q) >= 3:
        yf_results = _yahoo_search(q)
        seen = {r["symbol"] for r in local}
        for r in yf_results:
            if r["symbol"] not in seen:
                local.append(r)
                seen.add(r["symbol"])

    return local[:20]


@app.get("/company/{symbol}")
def company(symbol: str):
    """Return key company metrics — direct Yahoo Finance API via curl_cffi."""
    symbol = symbol.upper().strip()
    cache_key = f"company:{symbol}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    ns = _get_yf_ticker(symbol)
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

    ns = _get_yf_ticker(symbol)
    # financialData gives TTM figures — used to fill FY gap when annual stmts lag
    data = _yf_summary(ns, "incomeStatementHistory,defaultKeyStatistics,financialData")
    stmts = (data.get("incomeStatementHistory") or {}).get("incomeStatementHistory") or []

    if not stmts:
        raise HTTPException(404, f"No financials for {ns}")

    stmts = sorted(stmts, key=lambda x: x.get("endDate", {}).get("raw", 0))[-5:]

    # Shares outstanding — used to calculate EPS when Yahoo doesn't provide it directly
    ks = data.get("defaultKeyStatistics") or {}
    fd = data.get("financialData") or {}
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

    # ------------------------------------------------------------------
    # Inject TTM row if annual stmts lag behind current Indian FY
    # Indian FY ends March 31. Yahoo often only has last-filed annual,
    # which can be 1-2 years old. financialData.totalRevenue = TTM.
    # ------------------------------------------------------------------
    today_dt = datetime.utcnow()
    # Indian FY year = calendar year of March 31 end
    # e.g. May 2026 → current FY = 2026 (April 2025–March 2026)
    current_fy = today_dt.year if today_dt.month >= 4 else today_dt.year - 1

    last_stmt_ts = (stmts[-1].get("endDate") or {}).get("raw", 0) if stmts else 0
    last_stmt_year = datetime.utcfromtimestamp(last_stmt_ts).year if last_stmt_ts else 0

    if last_stmt_year < current_fy and fd:
        ttm_revenue = safe_val(fd.get("totalRevenue", {}).get("raw") if isinstance(fd.get("totalRevenue"), dict) else fd.get("totalRevenue"))
        ttm_ebitda  = safe_val(fd.get("ebitda", {}).get("raw") if isinstance(fd.get("ebitda"), dict) else fd.get("ebitda"))
        ttm_pat_raw = fd.get("netIncomeToCommon") or fd.get("netIncome")
        ttm_pat     = safe_val(ttm_pat_raw.get("raw") if isinstance(ttm_pat_raw, dict) else ttm_pat_raw)
        # Fallback: derive PAT from trailingEps × shares
        if not ttm_pat and trailing_eps and shares_outstanding:
            ttm_pat = trailing_eps * shares_outstanding

        if ttm_revenue:
            ttm_label      = f"FY{str(current_fy)[2:]}"
            ttm_revenue_cr = round(ttm_revenue / 1e7, 2)
            ttm_pat_cr     = round(ttm_pat / 1e7, 2) if ttm_pat else 0.0
            ttm_ebitda_cr  = round(ttm_ebitda / 1e7, 2) if ttm_ebitda else 0.0
            ttm_net_margin = round((ttm_pat / ttm_revenue) * 100, 2) if ttm_pat and ttm_revenue else 0.0
            ttm_ebitda_margin = round((ttm_ebitda / ttm_revenue) * 100, 2) if ttm_ebitda and ttm_revenue else 0.0
            ttm_rev_growth = 0.0
            if prev_revenue and prev_revenue > 0 and ttm_revenue:
                ttm_rev_growth = round(((ttm_revenue - prev_revenue) / prev_revenue) * 100, 2)

            rows.append({
                "year": ttm_label,
                "revenue": ttm_revenue_cr,
                "pat": ttm_pat_cr,
                "ebitda": ttm_ebitda_cr,
                "eps": round(trailing_eps, 2) if trailing_eps else 0.0,
                "netMargin": ttm_net_margin,
                "revenueGrowth": ttm_rev_growth,
                "ebitdaMargin": ttm_ebitda_margin,
            })
            rows = rows[-5:]  # keep only 5 years

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


@app.get("/historical/{symbol}")
def historical_valuation(symbol: str):
    """
    Return 5Y monthly price history + reconstructed P/E and P/B series.
    Used by the Historical Valuation Chart on the frontend.

    Logic:
      - Fetch monthly OHLCV from Yahoo Finance chart API
      - Fetch annual EPS from incomeStatementHistory
      - Fetch annual BVPS from balanceSheetHistory
      - For each monthly price, find trailing EPS/BVPS (most recent FY that ended before that month)
      - Return points array + percentile stats
    """
    symbol = symbol.upper().strip()
    cache_key = f"historical:{symbol}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    ns = _get_yf_ticker(symbol)
    _ensure_yahoo()
    if not _YF_SESSION_OBJ or not _YF_CRUMB:
        raise HTTPException(503, "Yahoo Finance session unavailable")

    # ── 1. Monthly price history (5Y) ──────────────────────────────────────
    try:
        chart_resp = _YF_SESSION_OBJ.get(
            f"{_YF_HOST}/v8/finance/chart/{ns}",
            params={"range": "5y", "interval": "1mo", "crumb": _YF_CRUMB},
            timeout=20,
        )
        if not chart_resp.ok:
            raise HTTPException(chart_resp.status_code, f"Chart API error for {ns}")
        chart_json   = chart_resp.json()
        chart_result = (chart_json.get("chart") or {}).get("result") or []
        if not chart_result:
            raise HTTPException(404, f"No chart data for {ns}")
        cr         = chart_result[0]
        timestamps = cr.get("timestamp") or []
        closes     = (cr.get("indicators") or {}).get("quote", [{}])[0].get("close") or []
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Chart fetch failed: {e}")

    if not timestamps or not closes:
        raise HTTPException(404, f"Empty chart data for {ns}")

    # ── 2. Annual financials (EPS + Book Value) ───────────────────────────
    try:
        fin_data = _yf_summary(ns, "incomeStatementHistory,balanceSheetHistory,defaultKeyStatistics")
    except Exception as e:
        raise HTTPException(502, f"Financials fetch failed: {e}")

    ks = fin_data.get("defaultKeyStatistics") or {}
    so_raw = ks.get("sharesOutstanding")
    shares = float(so_raw.get("raw", 0) if isinstance(so_raw, dict) else (so_raw or 0))

    def rv(d: dict, key: str) -> float:
        v = d.get(key)
        if isinstance(v, dict): return safe_val(v.get("raw"))
        return safe_val(v)

    # Build EPS timeline: sorted list of (end_timestamp, eps_per_share)
    eps_timeline: list[tuple[float, float]] = []
    stmts = (fin_data.get("incomeStatementHistory") or {}).get("incomeStatementHistory") or []
    for stmt in stmts:
        end_ts  = (stmt.get("endDate") or {}).get("raw", 0)
        pat     = rv(stmt, "netIncome")
        eps_raw = rv(stmt, "basicEPS") or rv(stmt, "dilutedEPS") or rv(stmt, "basicEps") or rv(stmt, "dilutedEps")
        if not eps_raw and pat and shares > 0:
            eps_raw = pat / shares
        if end_ts and eps_raw:
            eps_timeline.append((float(end_ts), float(eps_raw)))
    eps_timeline.sort(key=lambda x: x[0])

    # Build BVPS timeline: sorted list of (end_timestamp, bvps)
    bvps_timeline: list[tuple[float, float]] = []
    bs_stmts = (fin_data.get("balanceSheetHistory") or {}).get("balanceSheetStatements") or []
    for bs in bs_stmts:
        end_ts = (bs.get("endDate") or {}).get("raw", 0)
        equity = rv(bs, "totalStockholderEquity")
        if end_ts and equity and shares > 0:
            bvps_timeline.append((float(end_ts), float(equity) / shares))
    bvps_timeline.sort(key=lambda x: x[0])

    def trailing_value(timeline: list[tuple[float, float]], price_ts: float) -> float | None:
        """Return the most recent value from timeline where end_ts <= price_ts."""
        result = None
        for end_ts, val in timeline:
            if end_ts <= price_ts + 86400 * 90:   # allow 90-day lag for results release
                result = val
        return result

    # ── 3. Build monthly data points ──────────────────────────────────────
    points = []
    for ts, price in zip(timestamps, closes):
        if price is None or price <= 0:
            continue
        date_str = datetime.utcfromtimestamp(ts).strftime("%Y-%m")
        eps  = trailing_value(eps_timeline,  float(ts))
        bvps = trailing_value(bvps_timeline, float(ts))

        pe_val = round(price / eps,  1)  if eps  and eps  > 0 else None
        pb_val = round(price / bvps, 2)  if bvps and bvps > 0 else None

        # Sanity cap — extreme outliers skew charts
        if pe_val and (pe_val < 0 or pe_val > 500):  pe_val  = None
        if pb_val and (pb_val < 0 or pb_val > 100):  pb_val  = None

        points.append({"date": date_str, "price": round(price, 2), "pe": pe_val, "pb": pb_val})

    if not points:
        raise HTTPException(404, "No valid historical data points")

    # ── 4. Statistics ─────────────────────────────────────────────────────
    def _stats(vals: list[float]) -> dict:
        if not vals:
            return {"min": 0, "max": 0, "median": 0, "p25": 0, "p75": 0, "mean": 0}
        sv   = sorted(vals)
        n    = len(sv)
        med  = sv[n // 2] if n % 2 else (sv[n // 2 - 1] + sv[n // 2]) / 2
        p25  = sv[int(n * 0.25)]
        p75  = sv[int(n * 0.75)]
        return {
            "min":    round(min(sv), 2),
            "max":    round(max(sv), 2),
            "median": round(med,     2),
            "p25":    round(p25,     2),
            "p75":    round(p75,     2),
            "mean":   round(sum(sv) / n, 2),
        }

    pe_vals  = [p["pe"]  for p in points if p["pe"]  is not None]
    pb_vals  = [p["pb"]  for p in points if p["pb"]  is not None]

    result = {
        "symbol": symbol,
        "points": points,
        "stats": {
            "pe": _stats(pe_vals),
            "pb": _stats(pb_vals),
        },
    }
    _cache_set(cache_key, result)
    return result


@app.get("/universe/size")
def universe_size():
    return {"count": len(STOCK_UNIVERSE), "source": "NSE EQUITY_L.csv"}
