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

import re
try:
    from bs4 import BeautifulSoup
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False

try:
    from curl_cffi import requests as cffi_requests
    _CURL_AVAILABLE = True
except ImportError:
    _CURL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Screener.in Integration
# ---------------------------------------------------------------------------
# Screener.in reads BSE/NSE regulatory filings directly — most accurate source
# for Indian stock fundamentals. Revenue is always in ₹ Crore. No unit guessing.
#
# Architecture:
#   Screener.in  → all fundamentals (P/E, P/B, ROE, revenue, PAT, EPS, OCF)
#   Yahoo Finance → live price, change%, 52w high/low, beta (it's good at these)
#   Yahoo Finance → historical price series for the valuation chart
#
# To enable authenticated access (gets more data, more reliable):
#   Set SCREENER_USERNAME and SCREENER_PASSWORD env vars in Render.
#   Free account at screener.in is sufficient.
#   Without creds: falls back to public page scraping (still works, less reliable).
# ---------------------------------------------------------------------------

SCREENER_USERNAME = os.environ.get("SCREENER_USERNAME", "")
SCREENER_PASSWORD = os.environ.get("SCREENER_PASSWORD", "")

_screener_session: Any = None          # requests.Session kept alive
_screener_session_at: float = 0        # unix timestamp when session was created
_SCREENER_SESSION_TTL = 18 * 3600      # reauth every 18 hours


def _make_browser_headers(referer: str = "https://www.screener.in/") -> dict:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": referer,
    }


def _get_screener_session():
    """
    Return an authenticated requests.Session for Screener.in.
    Logs in once, caches session for 18 hours. Returns None if no credentials.
    """
    global _screener_session, _screener_session_at

    if not SCREENER_USERNAME or not SCREENER_PASSWORD:
        return None  # No creds → caller uses public page scraping

    # Return cached session if still fresh
    if _screener_session and (time.time() - _screener_session_at) < _SCREENER_SESSION_TTL:
        return _screener_session

    try:
        sess = requests.Session()
        sess.headers.update(_make_browser_headers())

        # Step 1: GET login page → extract Django CSRF token
        login_page = sess.get("https://www.screener.in/login/", timeout=15)
        if not _BS4_AVAILABLE:
            print("[Screener] BeautifulSoup not available — cannot authenticate")
            return None

        soup = BeautifulSoup(login_page.text, "lxml")
        csrf_el = soup.find("input", {"name": "csrfmiddlewaretoken"})
        if not csrf_el:
            print("[Screener] CSRF token not found on login page")
            return None
        csrf_token = csrf_el.get("value", "")

        # Step 2: POST credentials
        resp = sess.post(
            "https://www.screener.in/login/",
            data={
                "username": SCREENER_USERNAME,
                "password": SCREENER_PASSWORD,
                "csrfmiddlewaretoken": csrf_token,
                "next": "/",
            },
            headers={"Referer": "https://www.screener.in/login/"},
            allow_redirects=True,
            timeout=15,
        )

        # Login succeeded if we're no longer on the login page
        if resp.status_code == 200 and "/login" not in resp.url:
            _screener_session = sess
            _screener_session_at = time.time()
            print(f"[Screener] Authenticated as {SCREENER_USERNAME}")
            return sess
        else:
            print(f"[Screener] Login failed — status {resp.status_code}, url {resp.url}")
            return None

    except Exception as e:
        print(f"[Screener] Auth error: {e}")
        return None


def _fetch_screener_page(symbol: str) -> Any:
    """
    Fetch and parse the Screener.in CONSOLIDATED company page.
    Returns a BeautifulSoup object or None.

    KEY RULE: We ONLY return consolidated data.
    - With auth (session): consolidated page is accessible → use it.
    - Without auth: consolidated page redirects to login.
      In that case we return None — caller falls back to Yahoo Finance
      (which always returns consolidated data for Indian stocks).
    - We NEVER fall back to standalone Screener data, because standalone
      revenue for conglomerates (Reliance, Tata Motors, Adani) is 3-6x
      lower than consolidated, producing completely wrong valuations.
    """
    if not _BS4_AVAILABLE:
        return None

    sess = _get_screener_session()

    # ── With authenticated session: try consolidated then standalone ──────────
    # Authenticated users see consolidated by default — standalone is only
    # shown when the company has no subsidiaries (same data either way).
    if sess:
        for variant in ["consolidated", ""]:
            url = f"https://www.screener.in/company/{symbol}/{variant}/"
            try:
                resp = sess.get(url, timeout=20)
                if resp.status_code != 200:
                    continue
                soup = BeautifulSoup(resp.text, "lxml")
                if soup.find(id="top-ratios") or soup.find(id="profit-loss"):
                    # Detect if we got standalone when consolidated exists
                    # (Screener shows a "Switch to Consolidated" link in this case)
                    standalone_warning = soup.find("a", string=lambda t: t and "consolidated" in t.lower())
                    if standalone_warning and variant == "":
                        # Standalone-only page for a company that has consolidated view
                        # Still return it — for standalone companies this IS the right data
                        pass
                    print(f"[Screener] {symbol}: loaded {variant or 'standalone'} page ✓")
                    return soup
            except Exception as e:
                print(f"[Screener] Auth fetch error {symbol} ({variant}): {e}")
                continue
        return None

    # ── Without auth: ONLY try consolidated ───────────────────────────────────
    # If consolidated requires login, we return None (caller uses Yahoo instead).
    # We deliberately skip standalone to avoid conglomerate revenue mismatch.
    url = f"https://www.screener.in/company/{symbol}/consolidated/"
    try:
        if _CURL_AVAILABLE:
            resp = cffi_requests.get(
                url, headers=_make_browser_headers(), timeout=20, impersonate="chrome124"
            )
        else:
            resp = requests.get(url, headers=_make_browser_headers(), timeout=20)

        if resp.status_code != 200:
            print(f"[Screener] {symbol}: consolidated HTTP {resp.status_code} without auth → Yahoo fallback")
            return None

        # Check if we were redirected to login page (Screener returns 200 on login redirect)
        if "/login" in resp.url or "login" in resp.url:
            print(f"[Screener] {symbol}: consolidated requires auth → Yahoo fallback")
            return None

        soup = BeautifulSoup(resp.text, "lxml")

        # If the page has a login form, we got the login page (not the company page)
        if soup.find("form", {"action": lambda a: a and "login" in a}):
            print(f"[Screener] {symbol}: got login page without auth → Yahoo fallback")
            return None

        if soup.find(id="top-ratios") or soup.find(id="profit-loss"):
            print(f"[Screener] {symbol}: public consolidated page loaded ✓")
            return soup

    except Exception as e:
        print(f"[Screener] No-auth fetch error {symbol}: {e}")

    return None


def _cr(val_str: str) -> float:
    """
    Parse a ₹ Crore value string from Screener.
    Handles: '1,25,432', '12,345.67', '1.25L', '-234', '(1,234)', '25.34 %'
    All Screener financials are already in ₹ Crore — no conversion needed.

    Indian accounting notation:
      (1,234)  → negative  → -1234   [loss-making PAT rows]
      25.34 %  → strip %   → 25.34   [OPM% rows]
    """
    try:
        v = val_str.strip()
        if not v or v in ("--", "-", ""):
            return 0.0

        # Detect parenthetical negatives: (1,234) → -1234
        is_negative = v.startswith("(") and v.endswith(")")
        if is_negative:
            v = v[1:-1]  # strip parens

        # Strip all formatting characters
        v = (v.replace(",", "")
              .replace("₹", "")
              .replace("Cr.", "")
              .replace("Cr", "")
              .replace("%", "")
              .strip())

        if not v:
            return 0.0

        # Handle Lakh Crore suffix (e.g. "1.25L" = 1,25,000 Cr)
        if v.endswith("L") or v.endswith("l"):
            result = float(v[:-1]) * 100000
        else:
            result = float(v)

        return -result if is_negative else result
    except Exception:
        return 0.0


def _parse_screener_ratios(soup: Any) -> dict:
    """
    Extract key ratios from the #top-ratios section.
    Returns dict with: currentPrice, marketCap, pe, pb, roe, dividendYield,
                       bookValue, roce, debtToEquity (derived), sector
    """
    ratios: dict = {}

    # ── Key ratios list ──────────────────────────────────────────────────────
    ratio_ul = soup.find(id="top-ratios")
    if ratio_ul:
        for li in ratio_ul.find_all("li"):
            spans = li.find_all("span")
            if len(spans) < 2:
                continue
            name  = spans[0].get_text(strip=True).lower()
            value = spans[-1].get_text(strip=True).replace(",", "").replace("₹", "").replace("%", "").replace("Cr.", "").replace("Cr", "").strip()

            try:
                if "market cap" in name:
                    ratios["marketCap"] = _cr(value)
                elif "current price" in name:
                    ratios["currentPrice"] = float(value)
                elif name in ("stock p/e", "p/e"):
                    ratios["pe"] = float(value)
                elif "book value" in name:
                    ratios["bookValue"] = float(value)
                elif "dividend yield" in name:
                    ratios["dividendYield"] = float(value)
                elif "roce" in name:
                    ratios["roce"] = float(value)
                elif "roe" in name and "roce" not in name:
                    ratios["roe"] = float(value)
                elif "debt / equity" in name or "debt/equity" in name:
                    ratios["debtToEquity"] = float(value)
            except (ValueError, TypeError):
                pass

    # Derive P/B from price / book value (more accurate than Yahoo)
    price = ratios.get("currentPrice", 0)
    bv    = ratios.get("bookValue", 0)
    if price > 0 and bv > 0:
        ratios["pb"] = round(price / bv, 2)

    # ── Sector from company info section ────────────────────────────────────
    about = soup.find(id="about") or soup.find(class_="company-info")
    if about:
        sector_el = about.find(text=re.compile(r"Sector", re.I))
        if sector_el:
            sector_parent = sector_el.parent
            if sector_parent:
                nxt = sector_parent.find_next_sibling()
                if nxt:
                    ratios["sector"] = nxt.get_text(strip=True)

    # Fallback: check meta description or title for sector hints
    if "sector" not in ratios:
        meta = soup.find("meta", {"name": "description"})
        if meta:
            content = meta.get("content", "")
            # Screener often puts sector in description
            m = re.search(r'in the ([A-Za-z &]+) sector', content)
            if m:
                ratios["sector"] = m.group(1).strip()

    return ratios


def _parse_screener_financials(soup: Any) -> list:
    """
    Parse the Profit & Loss and Cash Flow tables from Screener.in.
    Returns list of FinancialYear dicts (last 5 years, newest last).
    All values already in ₹ Crore — no conversion needed.
    """

    def _parse_table(section_id: str):
        """Extract years + row data from a Screener data table."""
        section = soup.find(id=section_id)
        if not section:
            return [], {}

        table = section.find("table")
        if not table:
            return [], {}

        # Year headers — "Mar 2024", "Mar 2023", ...
        years = []
        thead = table.find("thead")
        if thead:
            for th in thead.find_all("th")[1:]:
                txt = th.get_text(strip=True)
                m = re.search(r"(\d{4})", txt)
                if m:
                    years.append(f"FY{m.group(1)[2:]}")
                elif "ttm" in txt.lower():
                    years.append("TTM")

        # Row data
        rows: dict = {}
        tbody = table.find("tbody")
        if tbody:
            for tr in tbody.find_all("tr"):
                cells = tr.find_all("td")
                if not cells:
                    continue
                label = cells[0].get_text(strip=True).lower().rstrip(" +").strip()
                vals  = []
                for td in cells[1: len(years) + 1]:
                    txt = td.get_text(strip=True)
                    vals.append(_cr(txt))
                rows[label] = vals

        return years, rows

    def _row(rows: dict, *keywords) -> list:
        """
        Find a row by label. Priority:
          1. Exact match — label == keyword
          2. Starts-with match — label starts with keyword (e.g. "sales " matches "sales")
          3. Contains match — keyword is a substring (broadest, used as last resort)

        This prevents "other income from sales" stealing the "sales" row.
        """
        # Pass 1: exact
        for kw in keywords:
            for label, vals in rows.items():
                if label == kw:
                    return vals
        # Pass 2: starts-with
        for kw in keywords:
            for label, vals in rows.items():
                if label.startswith(kw):
                    return vals
        # Pass 3: contains (substring fallback)
        for kw in keywords:
            for label, vals in rows.items():
                if kw in label:
                    return vals
        return []

    pl_years, pl_rows = _parse_table("profit-loss")
    cf_years, cf_rows = _parse_table("cash-flow")

    if not pl_years:
        return []

    revenues = _row(pl_rows, "sales", "revenue", "net interest income", "total income")
    pats     = _row(pl_rows, "net profit", "profit after tax", "pat")
    epss     = _row(pl_rows, "eps in rs", "eps (in rs)", "eps", "earning per share")
    ebitdas  = _row(pl_rows, "operating profit")
    opm_pcts = _row(pl_rows, "opm %", "opm%", "operating profit margin")
    ocfs     = _row(cf_rows, "cash from operating", "operating activities", "net cash from operating")

    result = []
    prev_rev: float = 0.0

    for i, yr in enumerate(pl_years):
        if yr == "TTM":
            continue  # TTM injected separately via Yahoo for freshness

        rev  = revenues[i] if i < len(revenues) else 0.0
        pat  = pats[i]     if i < len(pats)     else 0.0
        eps  = epss[i]     if i < len(epss)      else 0.0
        ebit = ebitdas[i]  if i < len(ebitdas)   else 0.0
        opm  = opm_pcts[i] if i < len(opm_pcts)  else 0.0
        ocf  = ocfs[i]     if i < len(ocfs)       else 0.0

        net_margin   = round((pat / rev) * 100, 2)  if rev > 0 else 0.0
        ebitda_pct   = opm if opm != 0 else (round((ebit / rev) * 100, 2) if rev > 0 else 0.0)
        rev_growth   = round(((rev - prev_rev) / prev_rev) * 100, 2) if prev_rev > 0 and rev > 0 else 0.0
        prev_rev     = rev if rev > 0 else prev_rev

        result.append({
            "year":          yr,
            "revenue":       round(rev,  2),
            "pat":           round(pat,  2),
            "ebitda":        round(ebit, 2),
            "eps":           round(eps,  2),
            "netMargin":     net_margin,
            "revenueGrowth": rev_growth,
            "ebitdaMargin":  round(ebitda_pct, 2),
            "ocf":           round(ocf,  2),   # Operating Cash Flow — not in Yahoo Finance
            "source":        "screener",
        })

    # Keep last 5 full years (drop older ones)
    annual = [r for r in result if not r["year"].startswith("TTM")]
    return annual[-5:]

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


@app.get("/company-v2/{symbol}")
def company_v2(symbol: str):
    """
    Company fundamentals — Screener.in primary, Yahoo Finance fallback.

    Screener gives: P/E, P/B, ROE, ROCE, Market Cap, Book Value, D/E from BSE filings.
    Yahoo gives:    live price, change%, 52-week high/low, beta, shares outstanding.
    Combined:       best-of-both response with same interface as /company.
    """
    symbol = symbol.upper().strip()
    cache_key = f"company_v2:{symbol}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    # ── Always fetch live price from Yahoo — it's reliable for this ──────────
    ns   = _get_yf_ticker(symbol)
    data = _yf_summary(ns, "price,defaultKeyStatistics,summaryDetail,assetProfile,financialData")

    pr = data.get("price", {})
    ks = data.get("defaultKeyStatistics", {})
    sd = data.get("summaryDetail", {})
    ap = data.get("assetProfile", {})
    fd = data.get("financialData", {})

    def rv(d: dict, key: str) -> Any:
        v = d.get(key)
        if isinstance(v, dict):
            return v.get("raw")
        return v

    price_val  = safe_val(rv(pr, "regularMarketPrice") or rv(fd, "currentPrice"))
    prev_close = safe_val(rv(pr, "regularMarketPreviousClose") or rv(sd, "previousClose"))
    change     = round(price_val - prev_close, 2) if prev_close else 0.0
    change_pct = round((change / prev_close) * 100, 2) if prev_close else 0.0
    shares_cr  = round(safe_val(rv(ks, "sharesOutstanding")) / 1e7, 2)
    beta       = safe_val(rv(ks, "beta"))
    w52_high   = safe_val(rv(sd, "fiftyTwoWeekHigh"))
    w52_low    = safe_val(rv(sd, "fiftyTwoWeekLow"))
    rev_g      = rv(fd, "revenueGrowth")
    ear_g      = rv(fd, "earningsGrowth")
    fwd_pe     = safe_val(rv(sd, "forwardPE"))

    if price_val == 0:
        raise HTTPException(404, f"No price data for {ns}")

    stock_meta = STOCK_UNIVERSE.get(symbol, {})

    # Build base result from Yahoo (always available)
    result = {
        "symbol":         symbol,
        "name":           rv(pr, "longName") or rv(pr, "shortName") or stock_meta.get("name", symbol),
        "sector":         ap.get("sector") or stock_meta.get("sector", "Unknown"),
        "industry":       ap.get("industry", ""),
        "currentPrice":   price_val,
        "previousClose":  prev_close,
        "change":         change,
        "changePct":      change_pct,
        "marketCap":      round(safe_val(rv(pr, "marketCap") or rv(sd, "marketCap")) / 1e7, 2),
        "pe":             safe_val(rv(sd, "trailingPE")),
        "forwardPE":      fwd_pe,
        "pb":             safe_val(rv(ks, "priceToBook")),
        "roe":            round(float(rv(fd, "returnOnEquity")) * 100, 2) if rv(fd, "returnOnEquity") else 0.0,
        "roa":            round(float(rv(fd, "returnOnAssets")) * 100, 2) if rv(fd, "returnOnAssets") else 0.0,
        "eps":            safe_val(rv(ks, "trailingEps")),
        "dividendYield":  round(float(rv(sd, "dividendYield")) * 100, 2) if rv(sd, "dividendYield") else 0.0,
        "week52High":     w52_high,
        "week52Low":      w52_low,
        "debtToEquity":   round(safe_val(rv(fd, "debtToEquity")) / 100, 2) if rv(fd, "debtToEquity") else 0.0,
        "currentRatio":   safe_val(rv(fd, "currentRatio")),
        "shares":         shares_cr,
        "beta":           beta,
        "revenueGrowth":  round(float(rev_g) * 100, 2) if rev_g else 0.0,
        "earningsGrowth": round(float(ear_g) * 100, 2) if ear_g else 0.0,
        "dataSource":     "yahoo",
    }

    # ── Overlay with Screener.in fundamentals (much more accurate) ────────────
    try:
        soup = _fetch_screener_page(symbol)
        if soup:
            ratios = _parse_screener_ratios(soup)

            # Screener values override Yahoo for these fields (higher accuracy)
            if ratios.get("pe",          0) > 0:  result["pe"]          = ratios["pe"]
            if ratios.get("pb",          0) > 0:  result["pb"]          = ratios["pb"]
            if ratios.get("roe",         0) > 0:  result["roe"]         = ratios["roe"]
            if ratios.get("roce",        0) > 0:  result["roce"]        = ratios["roce"]
            if ratios.get("marketCap",   0) > 0:  result["marketCap"]   = ratios["marketCap"]
            if ratios.get("bookValue",   0) > 0:  result["bookValue"]   = ratios["bookValue"]
            if ratios.get("dividendYield",0) >= 0: result["dividendYield"] = ratios.get("dividendYield", result["dividendYield"])
            if ratios.get("debtToEquity",0) > 0:  result["debtToEquity"]  = ratios["debtToEquity"]
            if ratios.get("sector",      ""):     result["sector"]       = ratios["sector"]

            result["dataSource"] = "screener+yahoo"
            print(f"[Screener] {symbol}: fundamentals loaded from Screener.in ✓")
        else:
            print(f"[Screener] {symbol}: page unavailable, using Yahoo fundamentals")

    except Exception as e:
        print(f"[Screener] {symbol} overlay error: {e}")

    _cache_set(cache_key, result)
    return result


@app.get("/financials-v2/{symbol}")
def financials_v2(symbol: str):
    """
    Financials from Screener.in (BeautifulSoup parsed) — revenue always in ₹ Crore.
    Includes OCF (Operating Cash Flow) which Yahoo Finance doesn't reliably provide.
    Falls back to /financials (Yahoo) if Screener is unavailable.
    """
    symbol = symbol.upper().strip()
    cache_key = f"financials_v2:{symbol}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    # ── Try Screener.in (BS4-parsed, reliable) ───────────────────────────────
    screener_data = None
    try:
        soup = _fetch_screener_page(symbol)
        if soup:
            screener_data = _parse_screener_financials(soup)
    except Exception as e:
        print(f"[Screener] financials_v2 error for {symbol}: {e}")

    if screener_data and len(screener_data) >= 2:
        # Attach shares from Yahoo (Screener doesn't expose shares outstanding directly)
        try:
            ns     = _get_yf_ticker(symbol)
            ks_raw = _yf_summary(ns, "defaultKeyStatistics").get("defaultKeyStatistics", {})
            so_raw = ks_raw.get("sharesOutstanding")
            shares_cr = float(
                so_raw.get("raw", 0) if isinstance(so_raw, dict) else (so_raw or 0)
            ) / 1e7
            for row in screener_data:
                row["shares"] = round(shares_cr, 2)
        except Exception:
            pass  # shares is optional — valuation models have fallbacks

        _cache_set(cache_key, screener_data)
        print(f"[Screener] {symbol}: {len(screener_data)} years from Screener.in ✓")
        return screener_data

    # ── Fallback: Yahoo Finance ──────────────────────────────────────────────
    print(f"[Screener] {symbol}: Screener unavailable, using Yahoo Finance")
    try:
        yahoo_data = financials(symbol)
        for row in yahoo_data:
            row["source"] = "yahoo"
        _cache_set(cache_key, yahoo_data)
        return yahoo_data
    except Exception as e:
        raise HTTPException(500, f"Both Screener and Yahoo failed for {symbol}: {e}")


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


# ---------------------------------------------------------------------------
# Sector peer map — curated Indian stocks by sector
# ---------------------------------------------------------------------------
_SECTOR_PEERS: dict[str, list[str]] = {
    "Information Technology": ["TCS.NS","INFY.NS","WIPRO.NS","HCLTECH.NS","TECHM.NS","LTIM.NS","MPHASIS.NS","PERSISTENT.NS","COFORGE.NS"],
    "Technology": ["TCS.NS","INFY.NS","WIPRO.NS","HCLTECH.NS","TECHM.NS","LTIM.NS","MPHASIS.NS","PERSISTENT.NS","COFORGE.NS"],
    "Banking": ["HDFCBANK.NS","ICICIBANK.NS","KOTAKBANK.NS","AXISBANK.NS","SBIN.NS","INDUSINDBK.NS","FEDERALBNK.NS","BANDHANBNK.NS","AUBANK.NS"],
    "Financial Services": ["HDFCBANK.NS","ICICIBANK.NS","KOTAKBANK.NS","AXISBANK.NS","SBIN.NS","BAJFINANCE.NS","BAJAJFINSV.NS","CHOLAFIN.NS"],
    "NBFC": ["BAJFINANCE.NS","BAJAJFINSV.NS","CHOLAFIN.NS","MUTHOOTFIN.NS","MANAPPURAM.NS","HDFCAMC.NS"],
    "FMCG": ["HINDUNILVR.NS","NESTLEIND.NS","BRITANNIA.NS","DABUR.NS","MARICO.NS","COLPAL.NS","ITC.NS","EMAMILTD.NS","GODREJCP.NS"],
    "Consumer": ["TITAN.NS","TRENT.NS","NYKAA.NS","DMART.NS","BATA.NS","RELAXO.NS","VMART.NS"],
    "Pharmaceuticals": ["SUNPHARMA.NS","DIVISLAB.NS","CIPLA.NS","DRREDDY.NS","AUROPHARMA.NS","TORNTPHARM.NS","ALKEM.NS","LALPATHLAB.NS"],
    "Healthcare": ["SUNPHARMA.NS","DIVISLAB.NS","CIPLA.NS","DRREDDY.NS","AUROPHARMA.NS","TORNTPHARM.NS","ALKEM.NS"],
    "Automobiles": ["TATAMOTORS.NS","MARUTI.NS","BAJAJ-AUTO.NS","EICHERMOT.NS","HEROMOTOCO.NS","MOTHERSON.NS","TVSMOTOR.NS","ASHOKLEY.NS"],
    "Energy": ["RELIANCE.NS","ONGC.NS","BPCL.NS","IOC.NS","GAIL.NS","PETRONET.NS","MGL.NS"],
    "Metals": ["TATASTEEL.NS","JSWSTEEL.NS","HINDALCO.NS","COALINDIA.NS","VEDL.NS","NMDC.NS","SAIL.NS","JINDALSTEL.NS"],
    "Infrastructure": ["LT.NS","SIEMENS.NS","ABB.NS","HAVELLS.NS","BHARTIARTL.NS","ADANIPORTS.NS","IRCTC.NS"],
    "Utilities": ["NTPC.NS","POWERGRID.NS","TATAPOWER.NS","TORNTPOWER.NS","ADANIGREEN.NS","CESC.NS"],
    "Telecom": ["BHARTIARTL.NS","IDEA.NS","TATACOMM.NS","HFCL.NS"],
    "Electronics": ["KAYNES.NS","DIXON.NS","AMBER.NS","SYRMA.NS","PG ELECTROPLAST.NS","BEL.NS"],
    "Conglomerate": ["RELIANCE.NS","ADANIENT.NS","ITC.NS","LT.NS","TATAMOTORS.NS","M&M.NS"],
    "Real Estate": ["DLF.NS","GODREJPROP.NS","PRESTIGE.NS","PHOENIXLTD.NS","OBEROI.NS","BRIGADE.NS"],
}


@app.get("/peers/{symbol}")
def get_peers(symbol: str):
    """
    Return sector peers with key financial metrics.
    Fetches metrics for up to 7 peers + the queried company itself.
    Used by the Peer Compare view on the frontend.
    """
    symbol = symbol.upper().strip()
    cache_key = f"peers:{symbol}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    ns = _get_yf_ticker(symbol)

    # ── 1. Get sector for this symbol ─────────────────────────────────────
    try:
        profile_data = _yf_summary(ns, "summaryProfile,price,summaryDetail,defaultKeyStatistics,financialData")
    except Exception as e:
        raise HTTPException(502, f"Cannot fetch profile for {symbol}: {e}")

    sector = (profile_data.get("summaryProfile") or {}).get("sector", "")
    # Fallback: check our universe map
    if not sector:
        sector = (STOCK_UNIVERSE.get(symbol) or {}).get("sector", "")

    # ── 2. Build peer list ────────────────────────────────────────────────
    peer_ns_list = _SECTOR_PEERS.get(sector, [])
    # Ensure self is included first; remove self from peers to avoid duplication
    self_in_list = any(p.lower() == ns.lower() for p in peer_ns_list)
    peers_only = [p for p in peer_ns_list if p.lower() != ns.lower()][:6]
    all_symbols = [ns] + peers_only  # self first, then up to 6 peers

    # ── 3. Extract metrics from a yf summary result ───────────────────────
    def _extract_metrics(data: dict, sym_ns: str, is_self: bool) -> dict | None:
        try:
            pr  = data.get("price") or {}
            ks  = data.get("defaultKeyStatistics") or {}
            fd  = data.get("financialData") or {}
            sd  = data.get("summaryDetail") or {}

            def rv(d: dict, key: str):
                v = d.get(key)
                if isinstance(v, dict): return v.get("raw")
                return v

            name = rv(pr, "shortName") or rv(pr, "longName") or sym_ns.replace(".NS","").replace(".BO","")
            ticker = sym_ns.replace(".NS","").replace(".BO","")

            mktcap_raw = rv(pr, "marketCap") or 0
            price_raw  = rv(pr, "regularMarketPrice") or 0

            pe = rv(sd, "trailingPE") or rv(ks, "trailingPE") or rv(ks, "forwardPE")
            pb = rv(ks, "priceToBook")
            ev_ebitda = rv(ks, "enterpriseToEbitda")
            rev_growth = (rv(fd, "revenueGrowth") or 0) * 100
            net_margin = (rv(fd, "profitMargins") or 0) * 100
            roe = (rv(fd, "returnOnEquity") or 0) * 100
            de  = rv(fd, "debtToEquity")

            # Sanity cap
            if pe and (pe > 500 or pe < 0): pe = None
            if pb and (pb > 100 or pb < 0): pb = None
            if ev_ebitda and (ev_ebitda > 200 or ev_ebitda < 0): ev_ebitda = None

            return {
                "symbol": ticker,
                "name": name,
                "marketCap": round(mktcap_raw / 1e7, 0) if mktcap_raw else None,  # in ₹ Cr
                "currentPrice": round(price_raw, 1) if price_raw else None,
                "pe": round(pe, 1) if pe else None,
                "pb": round(pb, 1) if pb else None,
                "evEbitda": round(ev_ebitda, 1) if ev_ebitda else None,
                "revenueGrowth": round(rev_growth, 1) if rev_growth else None,
                "netMargin": round(net_margin, 1) if net_margin else None,
                "roe": round(roe, 1) if roe else None,
                "de": round(de, 1) if de else None,
                "isSelf": is_self,
            }
        except Exception:
            return None

    # ── 4. Fetch all symbols ──────────────────────────────────────────────
    results = []
    for sym_ns in all_symbols:
        is_self = sym_ns.lower() == ns.lower()
        try:
            if is_self:
                data = profile_data  # already fetched
            else:
                time.sleep(0.25)  # small delay to avoid rate limiting
                data = _yf_summary(sym_ns, "price,summaryDetail,defaultKeyStatistics,financialData")
            row = _extract_metrics(data, sym_ns, is_self)
            if row:
                results.append(row)
        except Exception:
            pass  # skip peers that fail

    result = {"sector": sector, "peers": results}
    _cache_set(cache_key, result)
    return result


@app.get("/universe/size")
def universe_size():
    return {"count": len(STOCK_UNIVERSE), "source": "NSE EQUITY_L.csv"}
