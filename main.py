"""
ROBU Data Server
FastAPI server providing Indian stock data via Screener.in.
Supports full NSE universe (~2000 stocks) downloaded from NSE on startup.

Run: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, HTMLResponse
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

import csv
import hashlib
import io
import difflib as _difflib

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

    BSE-only stocks: Screener.in uses the same symbol as NSE for dual-listed stocks.
    For BSE-only stocks, it uses the company's BSE ticker. We try the symbol as-is,
    and also check the STOCK_UNIVERSE for a known BSE ticker as fallback.
    """
    if not _BS4_AVAILABLE:
        return None

    # Build list of symbol candidates to try on Screener
    # For most stocks the NSE symbol works. For BSE-only stocks, Screener uses
    # the same symbol or sometimes the BSE code. We try both.
    candidates = [symbol]
    bse_info = STOCK_UNIVERSE.get(symbol, {})
    bse_code = bse_info.get("bse_code") or bse_info.get("bseCode")
    if bse_code and str(bse_code) != symbol:
        candidates.append(str(bse_code))

    sess = _get_screener_session()

    # ── With authenticated session: try consolidated then standalone ──────────
    # Authenticated users see consolidated by default — standalone is only
    # shown when the company has no subsidiaries (same data either way).
    if sess:
        for sym_candidate in candidates:
            for variant in ["consolidated", ""]:
                url = f"https://www.screener.in/company/{sym_candidate}/{variant}/"
                try:
                    resp = sess.get(url, timeout=20)
                    if resp.status_code == 404:
                        break  # This symbol doesn't exist on Screener, try next candidate
                    if resp.status_code != 200:
                        continue
                    soup = BeautifulSoup(resp.text, "lxml")
                    if soup.find(id="top-ratios") or soup.find(id="profit-loss"):
                        print(f"[Screener] {symbol}→{sym_candidate}: loaded {variant or 'standalone'} page ✓")
                        return soup
                except Exception as e:
                    print(f"[Screener] Auth fetch error {sym_candidate} ({variant}): {e}")
                    continue
        return None

    # ── Without auth: ONLY try consolidated ───────────────────────────────────
    # If consolidated requires login, we return None (caller uses Yahoo instead).
    # We deliberately skip standalone to avoid conglomerate revenue mismatch.
    for sym_candidate in candidates:
        url = f"https://www.screener.in/company/{sym_candidate}/consolidated/"
        try:
            if _CURL_AVAILABLE:
                resp = cffi_requests.get(
                    url, headers=_make_browser_headers(), timeout=20, impersonate="chrome124"
                )
            else:
                resp = requests.get(url, headers=_make_browser_headers(), timeout=20)

            if resp.status_code == 404:
                continue  # Try next candidate
            if resp.status_code != 200:
                print(f"[Screener] {sym_candidate}: consolidated HTTP {resp.status_code} without auth → Yahoo fallback")
                continue

            # Check if we were redirected to login page (Screener returns 200 on login redirect)
            if "/login" in resp.url or "login" in resp.url:
                print(f"[Screener] {sym_candidate}: consolidated requires auth → Yahoo fallback")
                return None  # Auth required — no point trying other candidates

            soup = BeautifulSoup(resp.text, "lxml")

            # If the page has a login form, we got the login page (not the company page)
            if soup.find("form", {"action": lambda a: a and "login" in a}):
                print(f"[Screener] {sym_candidate}: got login page without auth → Yahoo fallback")
                return None  # Auth required globally

            if soup.find(id="top-ratios") or soup.find(id="profit-loss"):
                print(f"[Screener] {symbol}→{sym_candidate}: public consolidated page loaded ✓")
                return soup

        except Exception as e:
            print(f"[Screener] No-auth fetch error {sym_candidate}: {e}")

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


def _parse_screener_about(soup: Any) -> str:
    """Extract the plain-English business description (Screener's 'About' block).
    Falls back to the page meta description. Returns '' if nothing usable."""
    try:
        # Screener renders the description inside the company profile block.
        el = (soup.select_one(".company-profile .about")
              or soup.select_one("div.about")
              or soup.select_one(".company-profile p"))
        if el:
            txt = el.get_text(" ", strip=True)
            txt = re.sub(r"\[+\s*\d+\s*\]+", "", txt)        # drop [1] / [[1]] citation markers
            txt = re.sub(r"\s*Read More\s*$", "", txt).strip()
            if len(txt) >= 40:
                return txt[:1200]
        meta = soup.find("meta", {"name": "description"})
        if meta and meta.get("content"):
            return str(meta["content"]).strip()[:1200]
    except Exception:
        pass
    return ""


def _parse_screener_ratios(soup: Any) -> dict:
    """
    Extract key ratios from Screener.in #top-ratios section.
    Returns: currentPrice, marketCap, pe, pb, roe, roce, dividendYield,
             bookValue, debtToEquity, week52High, week52Low, sector, industry
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
            raw   = spans[-1].get_text(strip=True)
            value = raw.replace(",", "").replace("₹", "").replace("%", "").replace("Cr.", "").replace("Cr", "").strip()

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
                elif any(p in name for p in ("debt / equity", "debt/equity", "debt to equity", "d/e", "borrowing")):
                    ratios["debtToEquity"] = float(value)
                elif any(p in name for p in ("high / low", "52 week", "52w h/l", "high/low", "wk h/l", "week h/l")):
                    # Screener formats: "3,480 / 1,278" or "₹3,480 / ₹1,278"
                    clean = raw.replace(",", "").replace("₹", "").replace("₹", "")
                    parts = clean.split("/")
                    if len(parts) == 2:
                        try:
                            h = float(parts[0].strip())
                            l = float(parts[1].strip())
                            if h > 0 and l > 0:
                                ratios["week52High"] = h
                                ratios["week52Low"]  = l
                        except (ValueError, TypeError):
                            pass
                elif "face value" in name or "face val" in name:
                    ratios["faceValue"] = float(value)
                elif "pledged" in name:
                    # "Pledged percentage" — promoter shares pledged as loan collateral
                    ratios["pledgedPct"] = float(value)
            except (ValueError, TypeError):
                pass

    # Derive P/B from price / book value (more accurate than Yahoo)
    price = ratios.get("currentPrice", 0)
    bv    = ratios.get("bookValue", 0)
    if price > 0 and bv > 0:
        ratios["pb"] = round(price / bv, 2)

    # Fallback: parse D/E from the detailed ratios table if not in top-ratios
    if "debtToEquity" not in ratios and soup:
        try:
            for table in soup.find_all("table"):
                for row in table.find_all("tr"):
                    cells = row.find_all("td")
                    if len(cells) >= 2:
                        label_text = cells[0].get_text(strip=True).lower()
                        if any(p in label_text for p in ("debt / equity", "debt/equity", "debt to equity")):
                            val_text = cells[1].get_text(strip=True).replace(",","").replace("₹","").strip()
                            ratios["debtToEquity"] = float(val_text)
                            break
        except Exception:
            pass

    # ── Industry from Screener's breadcrumb / tags (more precise than Yahoo) ─
    # Screener shows the industry category in breadcrumb or tag links.
    # e.g.: /screens/industry/commodity-exchanges/ → "Commodity Exchanges"
    def _looks_like_domain(s: str) -> bool:
        """Return True if string looks like a website domain (e.g. 'ril.com', 'tata.com')."""
        import re as _re
        # Has a dot, no spaces, ends in common TLD
        return bool(_re.search(r'^[a-zA-Z0-9\-]+\.[a-zA-Z]{2,6}$', s.strip()))

    try:
        for a in soup.find_all("a", href=True):
            href = a.get("href", "")
            if "/screens/industry/" in href or "/screens/sector/" in href:
                label = a.get_text(strip=True)
                if label and len(label) > 2 and not _looks_like_domain(label):
                    ratios["screenerIndustry"] = label
                    break
        # Also check the company info section for sector/industry links
        # IMPORTANT: skip website domains (e.g. "ril.com" appears in company-links)
        company_info = soup.find(class_="company-links") or soup.find(class_="sub-links")
        if company_info and "screenerIndustry" not in ratios:
            for a in company_info.find_all("a", href=True):
                href = a.get("href", "")
                # Only use links that point to Screener industry/sector pages
                if "/screens/industry/" in href or "/screens/sector/" in href:
                    label = a.get_text(strip=True)
                    if label and len(label) > 3 and not _looks_like_domain(label):
                        ratios["screenerIndustry"] = label
                        break
    except Exception:
        pass

    # ── Sector from company info section ────────────────────────────────────
    about = soup.find(id="about") or soup.find(class_="company-info")
    if about:
        sector_el = about.find(text=re.compile(r"Sector", re.I))
        if sector_el:
            sector_parent = sector_el.parent
            if sector_parent:
                nxt = sector_parent.find_next_sibling()
                if nxt:
                    raw_sector = nxt.get_text(strip=True)
                    if raw_sector and not _looks_like_domain(raw_sector):
                        ratios["sector"] = raw_sector

    # Fallback: meta description
    if "sector" not in ratios:
        meta = soup.find("meta", {"name": "description"})
        if meta:
            content = meta.get("content", "")
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
    bs_years, bs_rows = _parse_table("balance-sheet")

    if not pl_years:
        return []

    revenues = _row(pl_rows, "sales", "revenue", "net interest income", "total income")
    pats     = _row(pl_rows, "net profit", "profit after tax", "pat")
    epss     = _row(pl_rows, "eps in rs", "eps (in rs)", "eps", "earning per share")
    ebitdas  = _row(pl_rows, "operating profit")
    opm_pcts = _row(pl_rows, "opm %", "opm%", "operating profit margin")
    ocfs     = _row(cf_rows, "cash from operating", "operating activities", "net cash from operating")
    interests = _row(pl_rows, "interest", "finance cost")

    # Balance-sheet rows — mapped by year label (BS years can differ from P&L years)
    borrow_vals  = _row(bs_rows, "borrowings", "total debt")
    eqcap_vals   = _row(bs_rows, "equity capital", "equity share capital", "share capital")
    reserve_vals = _row(bs_rows, "reserves")
    bs_borrow = {y: (borrow_vals[i] if i < len(borrow_vals) else 0.0) for i, y in enumerate(bs_years)}
    bs_equity = {
        y: (eqcap_vals[i] if i < len(eqcap_vals) else 0.0)
           + (reserve_vals[i] if i < len(reserve_vals) else 0.0)
        for i, y in enumerate(bs_years)
    }

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
        intr = interests[i] if i < len(interests) else 0.0

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
            "interest":      round(intr, 2),   # Interest expense — for coverage ratio
            "borrowings":    round(bs_borrow.get(yr, 0.0), 2),  # Total debt from balance sheet
            "equity":        round(bs_equity.get(yr, 0.0), 2),  # Equity capital + reserves
            "source":        "screener",
        })

    # Keep last 10 full years (Screener.in provides up to 10 years)
    annual = [r for r in result if not r["year"].startswith("TTM")]
    return annual[-10:]

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

# Corporate action redirects — {old_symbol: new_symbol}
# Applied at the start of all data endpoints so legacy symbols resolve automatically.
_SYMBOL_REDIRECTS: dict[str, str] = {
    # Nov 2025: Tata Motors demerged. TATAMOTORS → TMPV (Passenger Vehicles, original ISIN)
    # + TMCV (Tata Motors Ltd, Commercial Vehicles).
    # TMPV carries ISIN INE155A01022 (the original TATAMOTORS ISIN).
    "TATAMOTORS": "TMPV",
}

_FALLBACK_STOCKS: dict[str, dict] = {
    "RELIANCE":   _nse("Reliance Industries Ltd",                "Energy",                   "RELIANCE"),
    "TCS":        _nse("Tata Consultancy Services Ltd",          "Information Technology",   "TCS"),
    "INFY":       _nse("Infosys Ltd",                            "Information Technology",   "INFY"),
    "HDFCBANK":   _nse("HDFC Bank Ltd",                          "Banking",                  "HDFCBANK"),
    "ICICIBANK":  _nse("ICICI Bank Ltd",                         "Banking",                  "ICICIBANK"),
    "WIPRO":      _nse("Wipro Ltd",                              "Information Technology",   "WIPRO"),
    "BAJFINANCE": _nse("Bajaj Finance Ltd",                      "NBFC",                     "BAJFINANCE"),
    "HINDUNILVR": _nse("Hindustan Unilever Ltd",                 "FMCG",                     "HINDUNILVR"),
    "ITC":        _nse("ITC Ltd",                                "FMCG",                     "ITC"),
    "KOTAKBANK":  _nse("Kotak Mahindra Bank Ltd",                "Banking",                  "KOTAKBANK"),
    "LT":         _nse("Larsen & Toubro Ltd",                    "Infrastructure",           "LT"),
    "AXISBANK":   _nse("Axis Bank Ltd",                          "Banking",                  "AXISBANK"),
    "ASIANPAINT": _nse("Asian Paints Ltd",                       "Consumer",                 "ASIANPAINT"),
    "MARUTI":     _nse("Maruti Suzuki India Ltd",                "Automobiles",              "MARUTI"),
    "TITAN":      _nse("Titan Company Ltd",                      "Consumer",                 "TITAN"),
    "SUNPHARMA":  _nse("Sun Pharmaceutical Industries Ltd",      "Pharmaceuticals",          "SUNPHARMA"),
    "HCLTECH":    _nse("HCL Technologies Ltd",                   "Information Technology",   "HCLTECH"),
    # TATAMOTORS demerged Nov 2025 → TMPV (PV business) + TMCV (CV business)
    "TMPV":       _nse("Tata Motors Passenger Vehicles Ltd",     "Automobiles",              "TMPV"),
    "TMCV":       _nse("Tata Motors Ltd",                        "Automobiles",              "TMCV"),
    "TATASTEEL":  _nse("Tata Steel Ltd",                         "Metals",                   "TATASTEEL"),
    "SBIN":       _nse("State Bank of India",                    "Banking",                  "SBIN"),
    "ADANIENT":   _nse("Adani Enterprises Ltd",                  "Conglomerate",             "ADANIENT"),
    "BHARTIARTL": _nse("Bharti Airtel Ltd",                      "Telecom",                  "BHARTIARTL"),
    "KAYNES":     _nse("Kaynes Technology India Ltd",            "Electronics",              "KAYNES"),
    "NTPC":       _nse("NTPC Ltd",                               "Utilities",                "NTPC"),
    "ONGC":       _nse("Oil & Natural Gas Corporation Ltd",      "Energy",                   "ONGC"),
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


_BSE_SEED_FILE = os.path.join(os.path.dirname(__file__), "bse_seed.json")


def _merge_bse_records(records: list[dict]) -> int:
    """Merge normalized BSE records {code,name,isin,sector,sid} into STOCK_UNIVERSE.
    Skips stocks already present via NSE (matched by ISIN). Returns count added."""
    global STOCK_UNIVERSE
    isin_to_nse = {
        info.get("isin", ""): sym
        for sym, info in STOCK_UNIVERSE.items()
        if info.get("isin")
    }
    added = 0
    for rec in records:
        code   = str(rec.get("code", "")).strip()
        name   = str(rec.get("name", "")).strip()
        isin   = str(rec.get("isin", "") or "").strip()
        sector = str(rec.get("sector", "") or "").strip() or "BSE Listed"
        sid    = str(rec.get("sid", "") or "").strip().upper()
        if not code or not name:
            continue
        # Already in NSE universe (matched by ISIN) — skip, NSE is primary
        if isin and isin in isin_to_nse:
            continue
        # BSE-only stock — prefer the alphanumeric scrip_id (searchable,
        # Screener-compatible), fall back to numeric code
        key = sid or code
        if key not in STOCK_UNIVERSE:
            STOCK_UNIVERSE[key] = {
                "name": name,
                "sector": sector,
                "exchange": "BSE",
                "isin": isin,
                "bseCode": code,
                "yf_ticker": f"{code}.BO",
            }
            added += 1
    return added


def _fetch_bse_live() -> list[dict] | None:
    """Fetch the live BSE equity list. Returns normalized records, or None on failure.
    BSE blocks many datacenter IPs, so this routinely fails in production — callers
    must fall back to the committed seed file (_BSE_SEED_FILE)."""
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
            print(f"[ROBU] BSE list HTTP {resp.status_code} — falling back to seed")
            return None
        items = resp.json()
        if not isinstance(items, list) or not items:
            print("[ROBU] BSE list empty/unexpected — falling back to seed")
            return None
        # Normalize to the same shape as the seed file
        records = []
        for it in items:
            records.append({
                "code":   str(it.get("SCRIP_CD",    it.get("Scripcode", ""))).strip(),
                "name":   str(it.get("Scrip_Name",  "")).strip(),
                "isin":   str(it.get("ISIN_NUMBER", it.get("ISIN_NO", "")) or "").strip(),
                "sector": str(it.get("INDUSTRY",    it.get("industry", "")) or "").strip() or "BSE Listed",
                "sid":    str(it.get("scrip_id",    "")).strip().upper(),
            })
        # Opportunistically refresh the committed seed so it stays current
        try:
            with open(_BSE_SEED_FILE, "w") as f:
                json.dump({"saved_at": datetime.now().isoformat(), "count": len(records), "stocks": records}, f, separators=(",", ":"))
        except Exception:
            pass
        return records
    except Exception as e:
        print(f"[ROBU] BSE live fetch failed: {e} — falling back to seed")
        return None


def _load_bse_seed() -> list[dict]:
    """Load the committed BSE seed file (guaranteed fallback that survives ephemeral
    production filesystems, since it ships in the repo)."""
    try:
        with open(_BSE_SEED_FILE) as f:
            data = json.load(f)
        return data.get("stocks", [])
    except Exception as e:
        print(f"[ROBU] BSE seed load failed: {e}")
        return []


def _load_bse_universe():
    """Add BSE-only stocks to the universe. Tries the live BSE API first; if that is
    blocked (common on cloud hosts), falls back to the committed seed file so BSE
    stocks are ALWAYS searchable."""
    records = _fetch_bse_live()
    source = "live API"
    if not records:
        records = _load_bse_seed()
        source = "committed seed"
    if not records:
        print("[ROBU] BSE universe: no data from live API or seed — BSE stocks unavailable")
        return
    added = _merge_bse_records(records)
    print(f"[ROBU] BSE universe merged: {added} BSE-only stocks added (source: {source})")


def _get_yf_ticker(symbol: str) -> str:
    """Return the correct Yahoo Finance ticker for a symbol (e.g. TCS→TCS.NS, 543652→543652.BO)."""
    info = STOCK_UNIVERSE.get(symbol, {})
    if info.get("yf_ticker"):
        return info["yf_ticker"]
    # Numeric-only symbols are BSE script codes
    if symbol.isdigit():
        return f"{symbol}.BO"
    return f"{symbol}.NS"


# In-memory exchange resolution cache: symbol → ("SYMBOL.NS" | "SYMBOL.BO", "NSE" | "BSE")
# This avoids the extra Yahoo probe on every request after the first lookup.
_exchange_cache: dict[str, tuple[str, str]] = {}


def _resolve_ticker(symbol: str) -> tuple[str, str]:
    """
    Return (ticker_string, exchange) for a symbol.
    Pure universe-based — does NOT call Yahoo Finance.
    Exchange detection order:
      1. Cached result from previous call
      2. Known BSE stock (numeric code, or yf_ticker ends in .BO, or exchange="BSE")
      3. Default → NSE (.NS)
    Returns: (ticker_string, "NSE" | "BSE")
    """
    if symbol in _exchange_cache:
        return _exchange_cache[symbol]

    initial = _get_yf_ticker(symbol)
    info    = STOCK_UNIVERSE.get(symbol, {})

    is_bse = (
        symbol.isdigit()
        or info.get("exchange", "").upper() == "BSE"
        or info.get("yf_ticker", "").endswith(".BO")
    )

    if is_bse:
        _exchange_cache[symbol] = (initial, "BSE")
        return initial, "BSE"

    _exchange_cache[symbol] = (initial, "NSE")
    return initial, "NSE"


# Load universe on startup
_load_nse_universe()
_load_bse_universe()


# ---------------------------------------------------------------------------
# Simple TTL cache — {key: (data, timestamp)}
# ---------------------------------------------------------------------------
_cache: dict[str, tuple[Any, float]] = {}
CACHE_TTL = 900  # 15 minutes


def _cache_get(key: str, ttl: float | None = None) -> Any | None:
    entry = _cache.get(key)
    if entry is None:
        return None
    data, ts = entry
    if time.time() - ts > (ttl if ttl is not None else CACHE_TTL):
        del _cache[key]
        return None
    return data


def _cache_set(key: str, data: Any) -> None:
    _cache[key] = (data, time.time())


# ---------------------------------------------------------------------------
# NSE Bhavcopy — official end-of-day price feed (free, no auth needed)
# ---------------------------------------------------------------------------
# NSE publishes a full Bhavcopy CSV every trading day at ~6pm IST.
# URL: https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_DDMMYYYY.csv
# Contains: SYMBOL, SERIES, PREV_CLOSE, OPEN, HIGH, LOW, CLOSE for all NSE stocks.
# We download once per day and cache in memory — single source for current price.
# ---------------------------------------------------------------------------

_BHAVCOPY: dict[str, dict] = {}   # symbol → {close, prevClose, open, high, low}
_BHAVCOPY_DATE: str = ""          # DDMMYYYY of the ACTUAL day whose file we loaded
_BHAVCOPY_LOCK = False            # simple flag to prevent concurrent downloads
_BHAVCOPY_LAST_TRY = 0.0          # epoch of last download attempt (throttle retries)


def _refresh_bhavcopy() -> None:
    """Download the most recent NSE Bhavcopy CSV and cache all prices."""
    global _BHAVCOPY, _BHAVCOPY_DATE, _BHAVCOPY_LOCK, _BHAVCOPY_LAST_TRY

    today = datetime.now()
    today_str = today.strftime("%d%m%Y")

    if _BHAVCOPY_DATE == today_str and _BHAVCOPY:
        return   # Already have TODAY'S file — nothing newer to fetch
    # We only have an older day's file (e.g. fetched before NSE published today's
    # ~6pm). Keep serving it but retry periodically so we pick up today's close
    # once it's available — without hammering NSE on every request.
    if _BHAVCOPY and (time.time() - _BHAVCOPY_LAST_TRY) < 1800:
        return
    if _BHAVCOPY_LOCK:
        return   # Another request is loading it

    _BHAVCOPY_LOCK = True
    _BHAVCOPY_LAST_TRY = time.time()
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Referer": "https://www.nseindia.com/",
        }
        # Try last 5 calendar days (handles weekends + holidays)
        for delta in range(5):
            d = today - timedelta(days=delta)
            ds = d.strftime("%d%m%Y")
            url = f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{ds}.csv"
            try:
                resp = requests.get(url, headers=headers, timeout=30)
                if resp.status_code != 200 or len(resp.content) < 5000:
                    continue

                new_cache: dict[str, dict] = {}
                lines = resp.text.strip().split("\n")
                if not lines:
                    continue

                # Parse header to detect column positions (handles format changes)
                header = [h.strip().upper() for h in lines[0].split(",")]

                def col(name: str, *aliases: str) -> int:
                    for candidate in (name,) + aliases:
                        if candidate in header:
                            return header.index(candidate)
                    return -1

                idx_sym    = col("SYMBOL")
                idx_series = col("SERIES")
                idx_close  = col("CLOSE_PRICE", "CLOSE")
                idx_prev   = col("PREV_CLOSE", "PREVCLOSE")
                idx_open   = col("OPEN_PRICE", "OPEN")
                idx_high   = col("HIGH_PRICE", "HIGH")
                idx_low    = col("LOW_PRICE", "LOW")

                if idx_sym < 0 or idx_close < 0:
                    continue  # Unrecognised format

                for line in lines[1:]:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) <= max(idx_sym, idx_close):
                        continue
                    series = parts[idx_series].strip() if idx_series >= 0 else "EQ"
                    if series not in ("EQ", "BE", "BZ", "SM"):
                        continue  # Skip F&O, currency etc.
                    sym = parts[idx_sym].strip().upper()
                    if not sym:
                        continue
                    try:
                        new_cache[sym] = {
                            "close":     float(parts[idx_close]) if idx_close >= 0 else 0,
                            "prevClose": float(parts[idx_prev])  if idx_prev  >= 0 else 0,
                            "open":      float(parts[idx_open])  if idx_open  >= 0 else 0,
                            "high":      float(parts[idx_high])  if idx_high  >= 0 else 0,
                            "low":       float(parts[idx_low])   if idx_low   >= 0 else 0,
                        }
                    except (ValueError, IndexError):
                        pass

                if len(new_cache) > 500:
                    _BHAVCOPY = new_cache
                    _BHAVCOPY_DATE = ds  # ACTUAL day loaded — if it's not today, we'll retry later
                    print(f"[Bhavcopy] Loaded {len(_BHAVCOPY)} stocks from NSE ({ds})")
                    return

            except Exception as e:
                print(f"[Bhavcopy] Error fetching {ds}: {e}")

        print("[Bhavcopy] Could not load data for last 5 trading days — will use Screener price")
    finally:
        _BHAVCOPY_LOCK = False


def _bhavcopy_price(symbol: str) -> dict | None:
    """Return Bhavcopy price dict for symbol, or None if unavailable."""
    _refresh_bhavcopy()
    bare = symbol.upper().replace(".NS", "").replace(".BO", "")
    return _BHAVCOPY.get(bare)


# ---------------------------------------------------------------------------
# BSE Bhavcopy — official EOD price feed for BSE-listed stocks
# ---------------------------------------------------------------------------
# NSE Bhavcopy never covers BSE-only stocks, and Yahoo/Screener frequently
# return junk for them (wrong scrip match → 8x-off price + garbage name).
# BSE publishes a daily "UDiFF" common-format bhavcopy CSV. We download it once
# per day and key every row by BOTH scrip code (FinInstrmId) and ticker
# (TckrSymb), so a lookup works whether we have the numeric code or the symbol.
# Columns: FinInstrmId, ISIN, TckrSymb, FinInstrmNm, OpnPric, HghPric, LwPric,
#          ClsPric, PrvsClsgPric, FinInstrmTp (STK = equity).
# URL: https://www.bseindia.com/download/BhavCopy/Equity/BhavCopy_BSE_CM_0_0_0_<YYYYMMDD>_F_0000.CSV
# ---------------------------------------------------------------------------

_BSE_BHAVCOPY: dict[str, dict] = {}   # code|symbol → {close, prevClose, open, high, low, name}
_BSE_BHAVCOPY_DATE: str = ""          # YYYYMMDD of the ACTUAL day whose file we loaded
_BSE_BHAVCOPY_LOCK = False
_BSE_BHAVCOPY_LAST_TRY = 0.0          # epoch of last download attempt (throttle retries)


def _refresh_bse_bhavcopy() -> None:
    """Download the most recent BSE UDiFF bhavcopy and cache all EOD prices."""
    global _BSE_BHAVCOPY, _BSE_BHAVCOPY_DATE, _BSE_BHAVCOPY_LOCK, _BSE_BHAVCOPY_LAST_TRY

    today = datetime.now()
    today_str = today.strftime("%Y%m%d")
    if _BSE_BHAVCOPY_DATE == today_str and _BSE_BHAVCOPY:
        return  # have today's file
    # Only have an older day's file — retry periodically (throttled) for today's.
    if _BSE_BHAVCOPY and (time.time() - _BSE_BHAVCOPY_LAST_TRY) < 1800:
        return
    if _BSE_BHAVCOPY_LOCK:
        return

    _BSE_BHAVCOPY_LOCK = True
    _BSE_BHAVCOPY_LAST_TRY = time.time()
    try:
        # BSE blocks many datacenter IPs — prime a browser-like session first.
        if _CURL_AVAILABLE:
            sess = cffi_requests.Session(impersonate="chrome120")
        else:
            sess = requests.Session()
            sess.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            })
        try:
            sess.get("https://www.bseindia.com", timeout=15)
        except Exception:
            pass

        # Try the last 7 calendar days (handles weekends + holidays)
        for delta in range(7):
            d = today - timedelta(days=delta)
            ds = d.strftime("%Y%m%d")
            url = (f"https://www.bseindia.com/download/BhavCopy/Equity/"
                   f"BhavCopy_BSE_CM_0_0_0_{ds}_F_0000.CSV")
            try:
                resp = sess.get(url, headers={
                    "Referer": "https://www.bseindia.com/market-data.html",
                    "Accept": "*/*",
                }, timeout=30)
                ctype = resp.headers.get("content-type", "").lower()
                # A real bhavcopy is ~800KB CSV; the "not found" page is small HTML.
                if resp.status_code != 200 or len(resp.content) < 50000 or "html" in ctype:
                    continue

                lines = resp.text.strip().split("\n")
                if len(lines) < 100:
                    continue
                header = [h.strip() for h in lines[0].split(",")]
                if "FinInstrmId" not in header or "ClsPric" not in header:
                    continue  # Unrecognised format

                def col(name: str) -> int:
                    return header.index(name) if name in header else -1

                i_code  = col("FinInstrmId")
                i_sym   = col("TckrSymb")
                i_type  = col("FinInstrmTp")
                i_name  = col("FinInstrmNm")
                i_open  = col("OpnPric")
                i_high  = col("HghPric")
                i_low   = col("LwPric")
                i_close = col("ClsPric")
                i_prev  = col("PrvsClsgPric")

                new_cache: dict[str, dict] = {}
                for line in lines[1:]:
                    p = line.split(",")
                    if len(p) <= max(i_code, i_close):
                        continue
                    if i_type >= 0 and p[i_type].strip().upper() != "STK":
                        continue  # equities only — skip derivatives etc.
                    code = p[i_code].strip()
                    sym  = p[i_sym].strip().upper() if i_sym >= 0 else ""
                    try:
                        def num(idx: int) -> float:
                            return float(p[idx]) if idx >= 0 and p[idx].strip() else 0.0
                        rec = {
                            "close":     num(i_close),
                            "prevClose": num(i_prev),
                            "open":      num(i_open),
                            "high":      num(i_high),
                            "low":       num(i_low),
                            "name":      p[i_name].strip() if i_name >= 0 else "",
                        }
                    except (ValueError, IndexError):
                        continue
                    if rec["close"] <= 0:
                        continue
                    if code:
                        new_cache[code] = rec
                    if sym:
                        new_cache[sym] = rec

                if len(new_cache) > 1000:
                    _BSE_BHAVCOPY = new_cache
                    _BSE_BHAVCOPY_DATE = ds  # ACTUAL day loaded — if not today, we'll retry later
                    print(f"[BSE Bhavcopy] Loaded {len(new_cache)} entries from BSE ({ds})")
                    return
            except Exception as e:
                print(f"[BSE Bhavcopy] Error fetching {ds}: {e}")

        print("[BSE Bhavcopy] Could not load data for last 7 days")
    finally:
        _BSE_BHAVCOPY_LOCK = False


def _bse_bhavcopy_price(symbol: str) -> dict | None:
    """Return BSE Bhavcopy price dict for a symbol, or None if unavailable.
    Resolves the BSE scrip code from the universe, then falls back to the bare
    symbol / numeric code so it works for sids (BONDADA) and codes (543971)."""
    _refresh_bse_bhavcopy()
    if not _BSE_BHAVCOPY:
        return None
    info = STOCK_UNIVERSE.get(symbol, {})
    bare = symbol.upper().replace(".BO", "").replace(".NS", "")
    candidates = [
        str(info.get("bseCode") or info.get("bse_code") or "").strip(),
        bare,
    ]
    for key in candidates:
        if key and key in _BSE_BHAVCOPY:
            return _BSE_BHAVCOPY[key]
    return None


def _company_from_bhav(symbol: str, bhav: dict, stock_meta: dict, source: str) -> dict:
    """Build a company-v2 record from an EOD bhavcopy row when fundamentals
    aren't available (Screener/Yahoo missing). Price + name are correct;
    ratio fields are 0 and the frontend degrades gracefully."""
    p  = bhav["close"]
    pc = bhav.get("prevClose", 0)
    chg     = round(p - pc, 2) if pc else 0.0
    chg_pct = round((chg / pc) * 100, 2) if pc else 0.0
    return {
        "symbol":        symbol,
        "name":          bhav.get("name") or stock_meta.get("name", symbol),
        "sector":        stock_meta.get("sector", "Unknown"),
        "industry":      "",
        "exchange":      stock_meta.get("exchange", "NSE"),
        "currentPrice":  round(p, 2),
        "previousClose": round(pc, 2),
        "change":        chg,
        "changePct":     chg_pct,
        "marketCap":     0.0,
        "pe":            0.0, "forwardPE": 0.0, "pb": 0.0,
        "roe":           0.0, "roa":       0.0, "roce": 0.0,
        "eps":           0.0, "dividendYield": 0.0,
        "week52High":    0.0, "week52Low": 0.0,
        "debtToEquity":  0.0, "bookValue": 0.0,
        "currentRatio":  0.0, "shares":    0.0,
        "beta":          0.0, "revenueGrowth": 0.0, "earningsGrowth": 0.0,
        "dataSource":    source,
    }


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


# Yahoo Finance session initialisation disabled — all data now via Screener.in
# _init_yahoo()  # <-- was here; removed to avoid startup delay on Railway


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "universe_size": len(STOCK_UNIVERSE)}


def _search_row(sym: str, info: dict, match: str = "contains") -> dict:
    """Return clean search result — only fields the frontend needs.
    `match` tags how the row was found: exact | starts | contains | fuzzy —
    the frontend uses this to render a 'Did you mean…' prompt for typos."""
    return {
        "symbol": sym,
        "name": info.get("name", sym),
        "sector": info.get("sector", ""),
        "exchange": info.get("exchange", "NSE"),
        "match": match,
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
               "WIPRO", "BAJFINANCE", "ITC", "SBIN", "TMPV", "TMCV"]
        return [_search_row(sym, STOCK_UNIVERSE[sym]) for sym in top if sym in STOCK_UNIVERSE]

    q_lower = q.lower()
    exact, starts, contains = [], [], []

    for sym, info in STOCK_UNIVERSE.items():
        sym_lower = sym.lower()
        name_lower = info.get("name", "").lower()
        if sym_lower == q_lower:
            exact.append(_search_row(sym, info, "exact"))
        elif sym_lower.startswith(q_lower) or name_lower.startswith(q_lower):
            starts.append(_search_row(sym, info, "starts"))
        elif q_lower in sym_lower or q_lower in name_lower:
            contains.append(_search_row(sym, info, "contains"))

    # NSE results first within each tier, then BSE
    def _rank(r): return 0 if r["exchange"] == "NSE" else 1
    exact.sort(key=_rank)
    starts.sort(key=_rank)
    contains.sort(key=_rank)
    local = exact + starts + contains

    # ── Fuzzy "did you mean" fallback ──────────────────────────────────────
    # When the query has no strong match (likely a typo: "relianse", "infoys"),
    # find the closest symbols/names so the frontend can prompt a correction.
    if len(local) < 6 and len(q_lower) >= 3:
        seen = {r["symbol"] for r in local}
        scored: list[tuple[float, dict]] = []
        for sym, info in STOCK_UNIVERSE.items():
            if sym in seen:
                continue
            sym_lower = sym.lower()
            name_lower = info.get("name", "").lower()
            # Similarity vs the symbol (weighted higher — a ticker match is the
            # strongest signal) and vs each word of the company name.
            sym_score  = _difflib.SequenceMatcher(None, q_lower, sym_lower).ratio()
            name_score = max((_difflib.SequenceMatcher(None, q_lower, w).ratio()
                              for w in name_lower.split()), default=0.0)
            score = max(sym_score, name_score * 0.92)
            if score >= 0.62:
                scored.append((score, _search_row(sym, info, "fuzzy")))
        # Highest similarity first; NSE before BSE on ties
        scored.sort(key=lambda t: (-t[0], _rank(t[1])))
        local.extend(r for _, r in scored[:8])

    return local[:20]



def _pick_ceo(officers):
    """Best-guess current CEO/MD name from Yahoo assetProfile.companyOfficers."""
    if not officers or not isinstance(officers, list):
        return ""
    for kw in ("chief executive", "ceo", "managing director", "chairman & md", " md"):
        for o in officers:
            if kw in str(o.get("title", "")).lower() and o.get("name"):
                return str(o.get("name")).strip()
    for o in officers:
        if o.get("name"):
            return str(o.get("name")).strip()
    return ""

@app.get("/company/{symbol}")
def company(symbol: str):
    """Return key company metrics — direct Yahoo Finance API via curl_cffi."""
    symbol = symbol.upper().strip()
    symbol = _SYMBOL_REDIRECTS.get(symbol, symbol)   # handle demergers/renames
    cache_key = f"company:{symbol}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    ns, exchange = _resolve_ticker(symbol)
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

    # Use Yahoo's exchangeName if available, otherwise our resolved exchange
    yf_exchange = rv(pr, "exchangeName") or rv(pr, "exchange") or ""
    if "NSE" in yf_exchange.upper():
        resolved_exchange = "NSE"
    elif "BSE" in yf_exchange.upper() or "BOM" in yf_exchange.upper():
        resolved_exchange = "BSE"
    else:
        resolved_exchange = exchange  # from _resolve_ticker

    result = {
        "symbol": symbol,
        "name": rv(pr, "longName") or rv(pr, "shortName") or stock_meta.get("name", symbol),
        "sector": ap.get("sector") or stock_meta.get("sector", "Unknown"),
        "industry": ap.get("industry", ""),
        "description": (ap.get("longBusinessSummary") or "").strip(),
        "ceo": _pick_ceo(ap.get("companyOfficers")),
        "website": ap.get("website", "") or "",
        "employees": ap.get("fullTimeEmployees") or 0,
        "exchange": resolved_exchange,
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
    symbol = _SYMBOL_REDIRECTS.get(symbol, symbol)   # handle demergers/renames
    cache_key = f"financials:{symbol}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    ns, _exch = _resolve_ticker(symbol)
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
            rows = rows[-10:]  # keep up to 10 years

    _cache_set(cache_key, rows)
    return rows


@app.get("/company-v2/{symbol}")
def company_v2(symbol: str):
    """
    Company fundamentals — Screener.in + NSE Bhavcopy. No Yahoo Finance.

    Sources:
      NSE Bhavcopy  → current price, previous close, change, change%  (official EOD)
      Screener.in   → all fundamentals: P/E, P/B, ROE, ROCE, Market Cap,
                       Book Value, D/E, Dividend Yield, 52W High/Low, sector
      Local universe → company name, exchange fallback
    """
    symbol = symbol.upper().strip()
    symbol = _SYMBOL_REDIRECTS.get(symbol, symbol)   # handle demergers/renames
    cache_key = f"company_v2:{symbol}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    stock_meta = STOCK_UNIVERSE.get(symbol, {})
    _ns, exchange = _resolve_ticker(symbol)

    # ── 1. Screener.in — all fundamentals ────────────────────────────────────
    soup = _fetch_screener_page(symbol)

    # BSE-only stocks: Screener's URL slug is the numeric scrip code, not the
    # alphanumeric sid (e.g. /company/543971/ works, /company/BONDADA/ 404s).
    # When the sid lookup misses, retry with the code so we get full, correct
    # fundamentals instead of falling through to the junk Yahoo path.
    if not soup and exchange == "BSE":
        code = str(stock_meta.get("bseCode") or stock_meta.get("bse_code") or "").strip()
        if code and code != symbol:
            print(f"[company-v2] {symbol}: retrying Screener via BSE code {code}")
            soup = _fetch_screener_page(code)

    if not soup:
        # Screener.in doesn't have this company (e.g. TATAMOTORS returns 404 on Screener).
        print(f"[company-v2] Screener returned None for {symbol} — using EOD bhavcopy / Yahoo fallback")

        # BSE-only stocks: the Yahoo fallback returns junk (wrong scrip match →
        # 8x-off price + garbage name like "543971.BO,0P0001RHEK,..."), so the
        # official BSE Bhavcopy is authoritative for price AND name here.
        if exchange == "BSE":
            bhav = _bse_bhavcopy_price(symbol) or _bhavcopy_price(symbol)
            if bhav and bhav.get("close", 0) > 0:
                result = _company_from_bhav(symbol, bhav, stock_meta, "bse-bhavcopy")
                _cache_set(cache_key, result)
                return result

        # NSE stocks (or BSE with no bhavcopy yet): Yahoo Finance gives full
        # fundamentals, then NSE/BSE Bhavcopy as a last-resort price-only record.
        try:
            return company(symbol)   # v1 Yahoo Finance path (handles its own caching)
        except Exception as yf_err:
            print(f"[company-v2] Yahoo also failed for {symbol}: {yf_err}")
            bhav = _bhavcopy_price(symbol) or _bse_bhavcopy_price(symbol)
            if not bhav or bhav.get("close", 0) == 0:
                raise HTTPException(404, f"No data for {symbol}: Screener not indexed, Yahoo: {yf_err}")
            result = _company_from_bhav(symbol, bhav, stock_meta, "bhavcopy+universe")
            _cache_set(cache_key, result)
            return result

    ratios = _parse_screener_ratios(soup)

    # Extract company name from Screener's <h1> or page title
    screener_name = ""
    h1 = soup.find("h1")
    if h1:
        screener_name = h1.get_text(strip=True)
    if not screener_name:
        title = soup.find("title")
        if title:
            screener_name = title.get_text(strip=True).split("|")[0].strip()

    # ── 2. Official EOD price + change ────────────────────────────────────────
    # BSE-only stocks → BSE Bhavcopy (NSE Bhavcopy never covers them, and
    # Screener's price for BSE scrips is frequently a wrong-scrip mismatch).
    # NSE stocks → NSE Bhavcopy.
    if exchange == "BSE":
        bhav = _bse_bhavcopy_price(symbol) or _bhavcopy_price(symbol)
    else:
        bhav = _bhavcopy_price(symbol)

    if bhav and bhav.get("close", 0) > 0:
        price_val  = bhav["close"]
        prev_close = bhav["prevClose"]
        change     = round(price_val - prev_close, 2) if prev_close else 0.0
        change_pct = round((change / prev_close) * 100, 2) if prev_close and prev_close > 0 else 0.0
        price_src  = "bhavcopy"
    else:
        # Fallback: use Screener's current price (15-min delayed but fine for research tool)
        price_val  = ratios.get("currentPrice", 0.0)
        prev_close = 0.0
        change     = 0.0
        change_pct = 0.0
        price_src  = "screener"

    if price_val == 0:
        raise HTTPException(404, f"No price data available for {symbol}")

    # ── 3. Derive fields from Screener data ───────────────────────────────────
    market_cap  = ratios.get("marketCap", 0.0)
    pe          = ratios.get("pe", 0.0)
    pb          = ratios.get("pb", 0.0)
    roe         = ratios.get("roe", 0.0)
    roce        = ratios.get("roce", 0.0)
    de          = ratios.get("debtToEquity", 0.0)

    # Parse the financial-year rows once — reused for D/E, EPS/PE and the
    # distress check below.
    try:
        fin_rows = _parse_screener_financials(soup)
    except Exception as fin_err:
        print(f"[company-v2] financials parse failed for {symbol}: {fin_err}")
        fin_rows = []
    latest_fin    = fin_rows[-1] if fin_rows else {}
    latest_pat    = latest_fin.get("pat", 0.0) or 0.0
    latest_equity = latest_fin.get("equity", 0.0) or 0.0
    last_eps      = latest_fin.get("eps", 0.0) or 0.0
    # Distressed = making losses OR negative net worth (owes more than it owns).
    # For such names a P/E built off a single good year is misleading.
    # Screener's reported Book Value is the most reliable negative-net-worth signal
    # (a distressed firm can still post a one-off positive PAT year). For such names
    # a P/E built off a single good year is misleading.
    distressed    = latest_pat < 0 or latest_equity < 0 or ratios.get("bookValue", 0.0) < 0

    # Screener's top-ratios box rarely includes D/E → compute from the
    # balance sheet: Borrowings ÷ (Equity Capital + Reserves), latest year.
    if not de:
        for row in reversed(fin_rows):
            borrow = row.get("borrowings", 0.0)
            equity = row.get("equity", 0.0)
            if equity > 0:
                de = round(borrow / equity, 2)
                break
    div_yield   = ratios.get("dividendYield", 0.0)
    book_value  = ratios.get("bookValue", 0.0)
    w52_high    = ratios.get("week52High", 0.0)
    w52_low     = ratios.get("week52Low", 0.0)

    # EPS: prefer the actual reported EPS row (can be negative for loss-makers);
    # fall back to price/PE only when no financial EPS is available.
    if last_eps:
        eps = round(last_eps, 2)
    elif pe and pe > 0:
        eps = round(price_val / pe, 2)
    else:
        eps = 0.0

    # P/E: recompute off the live price + real EPS so it matches today's price.
    # NEVER manufacture a positive P/E for a distressed (loss-making /
    # negative-net-worth) company — leave it blank so the frontend honestly
    # shows "loss-making" instead of a misleadingly cheap multiple.
    if distressed:
        pe = 0.0
    elif last_eps > 0 and price_val > 0:
        pe = round(price_val / last_eps, 1)

    # Sector: prefer Screener's own industry tag, then screener sector, then universe
    sector   = ratios.get("screenerIndustry") or ratios.get("sector") or stock_meta.get("sector", "Unknown")
    industry = ratios.get("screenerIndustry", "")

    name = (
        screener_name
        or (bhav.get("name") if bhav else "")
        or stock_meta.get("name", "")
        or symbol
    )

    # Plain-English business description for the "About the company" card.
    description = _parse_screener_about(soup)

    result = {
        "symbol":         symbol,
        "name":           name,
        "description":    description,
        "sector":         sector,
        "industry":       industry,
        # Use the exchange resolved up-front (handles BSE-only scrips correctly);
        # don't re-derive it from a possibly-thin universe record.
        "exchange":       exchange,
        "currentPrice":   round(price_val, 2),
        "previousClose":  round(prev_close, 2),
        "change":         change,
        "changePct":      change_pct,
        "marketCap":      round(market_cap, 2),
        "pe":             pe,
        "forwardPE":      0.0,    # Screener doesn't publish forward PE
        "pb":             pb,
        "roe":            roe,
        "roa":            0.0,    # Screener doesn't show ROA directly
        "roce":           roce,
        "eps":            eps,
        "dividendYield":  div_yield,
        "week52High":     w52_high,
        "week52Low":      w52_low,
        "debtToEquity":   de,
        "bookValue":      book_value,
        "currentRatio":   0.0,    # Not on Screener top-ratios; available in balance sheet
        "shares":         round(market_cap / price_val, 2) if price_val > 0 and market_cap > 0 else 0.0,
        "beta":           0.0,    # Not needed for a valuation tool
        "revenueGrowth":  0.0,    # Computed from financials-v2 instead
        "earningsGrowth": 0.0,
        "pledgedPct":     ratios.get("pledgedPct"),  # Promoter pledge % (null = not published on page)
        "dataSource":     f"screener+{price_src}",
    }

    print(f"[company-v2] {symbol}: price={price_val} ({price_src}), PE={pe}, ROE={roe} ✓")
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
    symbol = _SYMBOL_REDIRECTS.get(symbol, symbol)   # handle demergers/renames
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

    if screener_data and len(screener_data) >= 1:
        # Compute shares from Market Cap / Price (no Yahoo needed)
        try:
            cv2 = _cache_get(f"company_v2:{symbol}")
            if cv2:
                mktcap = float(cv2.get("marketCap", 0) or 0)   # in ₹ Cr
                price  = float(cv2.get("currentPrice", 0) or 0)
                if mktcap > 0 and price > 0:
                    shares_cr = round(mktcap / price, 2)        # shares in Cr
                    for row in screener_data:
                        row["shares"] = shares_cr
        except Exception:
            pass  # shares is optional — valuation models have fallbacks

        _cache_set(cache_key, screener_data)
        print(f"[Screener] {symbol}: {len(screener_data)} years from Screener.in ✓")
        return screener_data

    # ── Fallback: Yahoo Finance income statements ────────────────────────────
    # Screener.in doesn't have this company — try Yahoo Finance (v1 path).
    # If Yahoo also fails, return [] so the frontend loads without crashing.
    print(f"[financials-v2] Screener failed for {symbol} — trying Yahoo Finance fallback")
    try:
        return financials(symbol)
    except Exception as yf_err:
        print(f"[financials-v2] Yahoo also failed for {symbol}: {yf_err} — returning empty financials")
        return []   # Empty list: company page loads but valuation charts are blank


_PRICE_TTL = 60  # seconds — keep portfolio/watchlist quotes near-live


def _yf_intraday_price(symbol: str) -> dict | None:
    """Live intraday quote from Yahoo Finance for NSE (.NS) or BSE (.BO).
    Returns {price, change, changePct, prevClose} or None. Works during market
    hours (delayed ~15m) AND for BSE-only stocks that NSE Bhavcopy never covers."""
    try:
        ns, _ = _resolve_ticker(symbol)
        pr = _yf_summary(ns, "price").get("price", {})
        p  = pr.get("regularMarketPrice")
        if p is None or float(p) <= 0:
            return None
        p  = float(p)
        pc = float(pr.get("regularMarketPreviousClose") or 0)
        chg = float(pr.get("regularMarketChange") or (p - pc if pc else 0))
        # Compute % ourselves — Yahoo's formatted=false percent is an inconsistent
        # fraction (0.0238 vs 2.38); deriving it from change/prevClose is reliable.
        chg_pct = (chg / pc * 100) if pc else 0.0
        return {"price": round(p, 2), "change": round(chg, 2),
                "changePct": round(chg_pct, 2), "prevClose": round(pc, 2)}
    except Exception:
        return None


@app.get("/price/{symbol}")
def price(symbol: str):
    """Current price — live Yahoo intraday first (NSE + BSE), then NSE Bhavcopy
    (official EOD), then Screener. 60-second cache keeps it near-live."""
    symbol = symbol.upper().strip()
    symbol = _SYMBOL_REDIRECTS.get(symbol, symbol)
    cache_key = f"price:{symbol}"
    cached = _cache_get(cache_key, ttl=_PRICE_TTL)
    if cached is not None:
        return cached

    _ns, exchange = _resolve_ticker(symbol)

    # 0) BSE-only stocks: official BSE Bhavcopy EOD close FIRST. Yahoo intraday
    #    returns junk for BSE scrips (wrong match → 8x-off price), and the user
    #    needs an accurate price to trade on, not a fast wrong one.
    if exchange == "BSE":
        bse = _bse_bhavcopy_price(symbol)
        if bse and bse.get("close", 0) > 0:
            p = bse["close"]; pc = bse.get("prevClose", 0)
            chg = round(p - pc, 2) if pc else 0.0
            chg_pct = round((chg / pc) * 100, 2) if pc else 0.0
            result = {"symbol": symbol, "price": p, "change": chg, "changePct": chg_pct,
                      "prevClose": pc, "source": "bse-bhavcopy"}
            _cache_set(cache_key, result)
            return result

    # 1) Live intraday (NSE; also BSE as a fallback if Bhavcopy missed)
    live = _yf_intraday_price(symbol)
    if live:
        result = {"symbol": symbol, **live, "source": "live"}
        _cache_set(cache_key, result)
        return result

    # 2) NSE Bhavcopy — official end-of-day close
    bhav = _bhavcopy_price(symbol)
    if bhav and bhav.get("close", 0) > 0:
        p = bhav["close"]
        pc = bhav.get("prevClose", 0)
        chg = round(p - pc, 2) if pc else 0.0
        chg_pct = round((chg / pc) * 100, 2) if pc else 0.0
        result = {"symbol": symbol, "price": p, "change": chg, "changePct": chg_pct,
                  "prevClose": pc, "source": "bhavcopy"}
        _cache_set(cache_key, result)
        return result

    # 3) Screener current price (last resort)
    soup = _fetch_screener_page(symbol)
    if soup:
        ratios = _parse_screener_ratios(soup)
        p = ratios.get("currentPrice", 0)
        if p:
            result = {"symbol": symbol, "price": p, "change": 0.0, "changePct": 0.0,
                      "prevClose": 0, "source": "screener"}
            _cache_set(cache_key, result)
            return result

    raise HTTPException(404, f"No price data for {symbol}")


# ---------------------------------------------------------------------------
# Screener.in Chart API — 5Y historical P/E, P/B, Price
# ---------------------------------------------------------------------------

def _get_screener_company_id(soup) -> str | None:
    """Extract Screener.in numeric company ID from the parsed company page.

    Tries multiple extraction methods in order. Screener uses the company ID in
    the chart API URL (/api/company/<ID>/chart/) and as data-id on several elements.
    """
    if not soup:
        return None

    # Method 1: data-id on any element (most reliable — Screener adds it to several divs)
    for el in soup.find_all(attrs={"data-id": True}):
        val = str(el.get("data-id", "")).strip()
        if val.isdigit() and len(val) >= 2:
            return val

    # Method 2: data-company-id attribute (alternate attribute name)
    for el in soup.find_all(attrs={"data-company-id": True}):
        val = str(el.get("data-company-id", "")).strip()
        if val.isdigit() and len(val) >= 2:
            return val

    # Method 3: chart API URL in anchor hrefs or src attributes
    # e.g. href="/api/company/12345/chart/" or src="/api/company/12345/chart/"
    for a in soup.find_all(href=True):
        m = re.search(r'/api/company/(\d+)/', a.get("href", ""))
        if m:
            return m.group(1)

    # Method 4: JavaScript variable in inline script tags
    for script in soup.find_all("script"):
        text = script.string or ""
        # Pattern: companyId: 12345  or  company_id = "12345"  or  "companyId":"12345"
        m = re.search(r'(?:companyId|company_id|"id"|\'id\')["\']?\s*[:=]\s*["\']?(\d{3,6})', text)
        if m:
            return m.group(1)
        # Also look for chart API URL in JS: url: '/api/company/12345/chart/'
        m = re.search(r'/api/company/(\d+)/chart', text)
        if m:
            return m.group(1)

    # Method 5: hidden input fields
    for inp in soup.find_all("input", {"type": "hidden"}):
        name = inp.get("name", "").lower()
        if "company" in name or name == "id":
            val = str(inp.get("value", "")).strip()
            if val.isdigit() and len(val) >= 2:
                return val

    # Method 6: canonical URL or og:url meta tag — Screener uses /company/<ID>/ format
    for meta in soup.find_all("meta"):
        content = meta.get("content", "")
        m = re.search(r'/company/(\d+)/', content)
        if m:
            return m.group(1)
    canonical = soup.find("link", {"rel": "canonical"})
    if canonical:
        m = re.search(r'/company/(\d+)/', canonical.get("href", ""))
        if m:
            return m.group(1)

    return None


def _normalize_screener_date(raw: str) -> str:
    """Convert Screener date formats (Jan 2024, 2024 Q1, 2024-01) to YYYY-MM."""
    raw = str(raw).strip()
    # YYYY-MM or YYYY-MM-DD
    m = re.match(r"^(\d{4})-(\d{2})", raw)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    # "Jan 2024" or "January 2024"
    months = {"jan":"01","feb":"02","mar":"03","apr":"04","may":"05","jun":"06",
              "jul":"07","aug":"08","sep":"09","oct":"10","nov":"11","dec":"12"}
    m = re.match(r"^([A-Za-z]{3})[a-z]*\s*(\d{4})", raw)
    if m:
        return f"{m.group(2)}-{months.get(m.group(1).lower(), '01')}"
    # "2024 Q1" → Indian FY: Q1=Jun, Q2=Sep, Q3=Dec, Q4=Mar
    m = re.match(r"^(\d{4})\s+Q(\d)", raw)
    if m:
        q_map = {"1": "06", "2": "09", "3": "12", "4": "03"}
        yr = int(m.group(1))
        q  = m.group(2)
        if q == "4":
            yr += 1
        return f"{yr}-{q_map.get(q, '06')}"
    # Unix ms timestamp
    if re.match(r"^\d{12,13}$", raw):
        try:
            return datetime.utcfromtimestamp(int(raw) // 1000).strftime("%Y-%m")
        except Exception:
            pass
    return raw[:7] if len(raw) >= 7 else raw


def _fetch_screener_chart_api(company_id: str, metric: str, days: int = 1825) -> list:
    """
    Call Screener.in chart API for historical metric time-series.
    Returns list of {"date": "YYYY-MM", "value": float}.
    Metrics: "Price", "Price-to-Earning", "Price-to-Book", "Return-on-Equity", "EPS"
    """
    sess = _get_screener_session()
    url  = f"https://www.screener.in/api/company/{company_id}/chart/"
    params = {"q": metric, "days": days, "consolidated": "true"}

    try:
        if sess:
            resp = sess.get(url, params=params, timeout=20)
        elif _CURL_AVAILABLE:
            resp = cffi_requests.get(url, params=params, headers=_make_browser_headers(), timeout=20, impersonate="chrome124")
        else:
            resp = requests.get(url, params=params, headers=_make_browser_headers(), timeout=20)

        if not resp.ok:
            print(f"[Screener chart] HTTP {resp.status_code} for {metric} (id={company_id})")
            return []

        data = resp.json()
        # Response: {"datasets": [{"label": "P/E", "data": [["Jan 2020", 22.5], ...]}]}
        datasets = data.get("datasets", [])
        if not datasets:
            return []

        points_raw = datasets[0].get("data", [])
        result = []
        for point in points_raw:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            date_raw, val = point[0], point[1]
            if val is None:
                continue
            try:
                result.append({"date": _normalize_screener_date(str(date_raw)), "value": float(val)})
            except (ValueError, TypeError):
                continue
        return result

    except Exception as e:
        print(f"[Screener chart] Error fetching {metric} for id={company_id}: {e}")
        return []


@app.get("/historical/{symbol}")
def historical_valuation(symbol: str):
    """
    5Y historical P/E, P/B and Price from Screener.in chart API.
    100% Screener — no Yahoo Finance dependency.
    """
    symbol = symbol.upper().strip()
    cache_key = f"historical:{symbol}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    # ── 1. Get Screener page + company ID ────────────────────────────────
    soup = _fetch_screener_page(symbol)
    if not soup:
        raise HTTPException(404, f"Company {symbol} not found on Screener.in")

    company_id = _get_screener_company_id(soup)
    if not company_id:
        raise HTTPException(500, f"Could not find Screener company ID for {symbol}")

    print(f"[historical] {symbol}: Screener company_id={company_id}")

    # ── 2. Fetch historical series from Screener chart API ────────────────
    pe_data    = _fetch_screener_chart_api(company_id, "Price-to-Earning", 1825)
    pb_data    = _fetch_screener_chart_api(company_id, "Price-to-Book",    1825)
    price_data = _fetch_screener_chart_api(company_id, "Price",            1825)

    # ── 3. Also parse annual EPS from Screener P&L to reconstruct P/E gaps ─
    # If chart API has gaps (e.g. stock was unlisted), fill from financials
    fin_data = _parse_screener_financials(soup)
    # Build EPS map: {"FY24": 45.2, ...}
    eps_by_fy = {row["year"]: row["eps"] for row in fin_data if row.get("eps", 0) > 0}

    # ── 4. Unify into monthly points ─────────────────────────────────────
    pe_map    = {d["date"]: d["value"] for d in pe_data}
    pb_map    = {d["date"]: d["value"] for d in pb_data}
    price_map = {d["date"]: d["value"] for d in price_data}

    all_dates = sorted(set(list(pe_map) + list(pb_map) + list(price_map)))

    points = []
    for date_str in all_dates:
        pe_val    = pe_map.get(date_str)
        pb_val    = pb_map.get(date_str)
        price_val = price_map.get(date_str)

        # Sanity caps
        if pe_val is not None and (pe_val < 0 or pe_val > 500):
            pe_val = None
        if pb_val is not None and (pb_val < 0 or pb_val > 100):
            pb_val = None

        if pe_val or pb_val or price_val:
            points.append({
                "date":  date_str,
                "price": round(price_val, 2) if price_val else None,
                "pe":    round(pe_val,    1) if pe_val    else None,
                "pb":    round(pb_val,    2) if pb_val    else None,
            })

    if not points:
        raise HTTPException(404, f"No historical data from Screener chart API for {symbol}")

    # ── 5. Statistics ─────────────────────────────────────────────────────
    def _stats(vals: list) -> dict:
        if not vals:
            return {"min": 0, "max": 0, "median": 0, "p25": 0, "p75": 0, "mean": 0}
        sv = sorted(vals)
        n  = len(sv)
        med = sv[n // 2] if n % 2 else (sv[n // 2 - 1] + sv[n // 2]) / 2
        return {
            "min":    round(min(sv), 2),
            "max":    round(max(sv), 2),
            "median": round(med, 2),
            "p25":    round(sv[int(n * 0.25)], 2),
            "p75":    round(sv[int(n * 0.75)], 2),
            "mean":   round(sum(sv) / n, 2),
        }

    pe_vals = [p["pe"] for p in points if p["pe"] is not None]
    pb_vals = [p["pb"] for p in points if p["pb"] is not None]

    result = {
        "symbol": symbol,
        "points": points,
        "stats": {
            "pe": _stats(pe_vals),
            "pb": _stats(pb_vals),
        },
        "source": "screener",
        "company_id": company_id,
    }
    _cache_set(cache_key, result)
    print(f"[historical] {symbol}: {len(points)} points, PE range {pe_vals[0] if pe_vals else '-'}–{pe_vals[-1] if pe_vals else '-'} ✓")
    return result


# ---------------------------------------------------------------------------
# Kite Connect (Zerodha) — reliable NSE/BSE candle data via the REST API.
# Uses plain HTTP (no kiteconnect SDK) to keep the dependency tree light.
# Requires env vars: KITE_API_KEY, KITE_API_SECRET
# Daily login: visit  https://<this-server>/kite/login  each morning
# (Kite access tokens expire ~7:30 AM IST, so one fresh login per day).
# ---------------------------------------------------------------------------
KITE_API_KEY     = os.environ.get("KITE_API_KEY", "")
KITE_API_SECRET  = os.environ.get("KITE_API_SECRET", "")
_KITE_HOST       = "https://api.kite.trade"
_KITE_TOKEN_FILE = "/tmp/kite_token.json"
_kite_access_token = ""
_kite_token_date   = ""
_kite_instruments: dict[str, int] = {}     # "TCS" -> instrument_token (NSE equity)
_kite_inst_ts = 0.0


def _get_kite_token() -> str:
    """Return today's access token from memory, the token file, or env."""
    global _kite_access_token, _kite_token_date
    today = time.strftime("%Y-%m-%d")
    if _kite_access_token and _kite_token_date == today:
        return _kite_access_token
    try:
        with open(_KITE_TOKEN_FILE) as f:
            d = json.load(f)
        if d.get("date") == today and d.get("access_token"):
            _kite_access_token, _kite_token_date = d["access_token"], today
            return _kite_access_token
    except Exception:
        pass
    env_tok = os.environ.get("KITE_ACCESS_TOKEN", "")
    if env_tok:
        _kite_access_token, _kite_token_date = env_tok, today
        return env_tok
    return ""


def _save_kite_token(tok: str) -> None:
    global _kite_access_token, _kite_token_date
    _kite_access_token = tok
    _kite_token_date = time.strftime("%Y-%m-%d")
    try:
        with open(_KITE_TOKEN_FILE, "w") as f:
            json.dump({"date": _kite_token_date, "access_token": tok}, f)
    except Exception:
        pass


def _kite_headers() -> dict | None:
    tok = _get_kite_token()
    if not KITE_API_KEY or not tok:
        return None
    return {"X-Kite-Version": "3", "Authorization": f"token {KITE_API_KEY}:{tok}"}


def _kite_instrument_token(symbol: str) -> int | None:
    """Map an NSE trading symbol (e.g. TCS) to its Kite instrument token."""
    global _kite_instruments, _kite_inst_ts
    headers = _kite_headers()
    if not headers:
        return None
    if not _kite_instruments or time.time() - _kite_inst_ts > 86400:
        try:
            resp = requests.get(f"{_KITE_HOST}/instruments/NSE", headers=headers, timeout=30)
            resp.raise_for_status()
            reader = csv.DictReader(io.StringIO(resp.text))
            mp: dict[str, int] = {}
            for row in reader:
                if row.get("segment") == "NSE":     # NSE = equity cash segment
                    mp[row["tradingsymbol"]] = int(row["instrument_token"])
            if mp:
                _kite_instruments = mp
                _kite_inst_ts = time.time()
                print(f"[kite] loaded {len(_kite_instruments)} NSE instruments")
        except Exception as e:
            print(f"[kite] instruments load failed: {e}")
            return None
    return _kite_instruments.get(symbol.upper())


def _kite_ohlc(symbol: str, period: str) -> list | None:
    """Daily candles from Kite. Returns None if Kite is unavailable/not logged in."""
    headers = _kite_headers()
    if not headers:
        return None
    token = _kite_instrument_token(symbol)
    if not token:
        return None
    days = {"6mo": 190, "1y": 370, "2y": 740, "5y": 1850}.get(period, 740)
    to_d = datetime.now()
    from_d = to_d - timedelta(days=days)
    url = f"{_KITE_HOST}/instruments/historical/{token}/day"
    params = {"from": from_d.strftime("%Y-%m-%d"), "to": to_d.strftime("%Y-%m-%d")}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=20)
        resp.raise_for_status()
        rows = (resp.json().get("data") or {}).get("candles") or []
    except Exception as e:
        print(f"[kite] historical failed for {symbol}: {e}")
        return None
    # each candle: [timestamp, open, high, low, close, volume]
    candles = [{
        "time":  str(c[0])[:10],
        "open":  round(float(c[1]), 2),
        "high":  round(float(c[2]), 2),
        "low":   round(float(c[3]), 2),
        "close": round(float(c[4]), 2),
    } for c in rows if len(c) >= 5]
    return candles or None


@app.get("/kite/login")
def kite_login():
    """Redirect the user to Kite's login page (do this once each morning)."""
    if not KITE_API_KEY:
        raise HTTPException(503, "Kite not configured — set KITE_API_KEY and KITE_API_SECRET.")
    return RedirectResponse(f"https://kite.zerodha.com/connect/login?v=3&api_key={KITE_API_KEY}")


@app.get("/kite/callback")
def kite_callback(request_token: str = "", status: str = ""):
    """Kite redirects here after login; exchange the request token for an access token."""
    if not request_token:
        raise HTTPException(400, "Missing request_token")
    checksum = hashlib.sha256(
        (KITE_API_KEY + request_token + KITE_API_SECRET).encode()
    ).hexdigest()
    try:
        resp = requests.post(
            f"{_KITE_HOST}/session/token",
            data={"api_key": KITE_API_KEY, "request_token": request_token, "checksum": checksum},
            headers={"X-Kite-Version": "3"},
            timeout=15,
        )
        resp.raise_for_status()
        tok = (resp.json().get("data") or {}).get("access_token")
    except Exception as e:
        raise HTTPException(400, f"Kite session exchange failed: {e}")
    if not tok:
        raise HTTPException(400, "Kite did not return an access token")
    _save_kite_token(tok)
    _kite_instruments.clear()  # force fresh instrument load with the new token
    return HTMLResponse(
        "<div style='font-family:system-ui;padding:40px;text-align:center'>"
        "<h2>✓ Kite connected</h2>"
        "<p>Live Kite data is active for today. You can close this tab.</p></div>"
    )


@app.get("/kite/status")
def kite_status():
    return {"configured": bool(KITE_API_KEY), "logged_in": bool(_get_kite_token())}


# ---------------------------------------------------------------------------
# Daily OHLC candles — powers the TradingView Lightweight price chart
# ---------------------------------------------------------------------------
def _yf_chart(symbol_ns: str, rng: str = "2y", interval: str = "1d") -> dict:
    """Call the Yahoo Finance chart API directly (daily OHLC time-series)."""
    _ensure_yahoo()
    if not _YF_SESSION_OBJ:
        raise HTTPException(503, "Yahoo Finance session unavailable on this server")

    params = {"range": rng, "interval": interval}
    resp = _YF_SESSION_OBJ.get(
        f"{_YF_HOST}/v8/finance/chart/{symbol_ns}", params=params, timeout=20
    )

    # Refresh session once if expired
    if resp.status_code in (401, 403):
        global _YF_SESSION_TS
        _YF_SESSION_TS = 0
        _init_yahoo()
        if not _YF_SESSION_OBJ:
            raise HTTPException(503, "Yahoo Finance session refresh failed")
        resp = _YF_SESSION_OBJ.get(
            f"{_YF_HOST}/v8/finance/chart/{symbol_ns}", params=params, timeout=20
        )

    if not resp.ok:
        raise HTTPException(resp.status_code, f"Yahoo chart error for {symbol_ns}: HTTP {resp.status_code}")

    data = resp.json()
    err = (data.get("chart") or {}).get("error")
    if err:
        raise HTTPException(404, (err or {}).get("description", "Not found"))
    results = (data.get("chart") or {}).get("result") or []
    if not results:
        raise HTTPException(404, f"No chart data for {symbol_ns}")
    return results[0]


@app.get("/ohlc/{symbol}")
def ohlc(symbol: str, period: str = "2y"):
    """
    Daily OHLC candles for the price chart.
    Tries the resolved exchange first, then falls back to the other suffix.
    """
    symbol = symbol.upper().strip()
    symbol = _SYMBOL_REDIRECTS.get(symbol, symbol)   # handle demergers/renames
    cache_key = f"ohlc:{symbol}:{period}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    # 1) Prefer Kite (Zerodha) — real exchange data, most reliable for NSE/BSE
    kite_candles = _kite_ohlc(symbol, period)
    if kite_candles:
        result = {"symbol": symbol, "candles": kite_candles, "source": "kite"}
        _cache_set(cache_key, result)
        print(f"[ohlc] {symbol}: {len(kite_candles)} candles (kite) ✓")
        return result

    # 2) Fall back to Yahoo Finance
    ticker, _ = _resolve_ticker(symbol)
    try:
        res = _yf_chart(ticker, rng=period)
    except HTTPException:
        alt = ticker.replace(".NS", ".BO") if ticker.endswith(".NS") else ticker.replace(".BO", ".NS")
        res = _yf_chart(alt, rng=period)

    timestamps = res.get("timestamp") or []
    quote = (((res.get("indicators") or {}).get("quote") or [{}])[0]) or {}
    opens  = quote.get("open")  or []
    highs  = quote.get("high")  or []
    lows   = quote.get("low")   or []
    closes = quote.get("close") or []

    candles = []
    for i, ts in enumerate(timestamps):
        o = opens[i]  if i < len(opens)  else None
        h = highs[i]  if i < len(highs)  else None
        l = lows[i]   if i < len(lows)   else None
        c = closes[i] if i < len(closes) else None
        if o is None or h is None or l is None or c is None:
            continue
        candles.append({
            "time":  time.strftime("%Y-%m-%d", time.gmtime(ts)),
            "open":  round(float(o), 2),
            "high":  round(float(h), 2),
            "low":   round(float(l), 2),
            "close": round(float(c), 2),
        })

    if not candles:
        raise HTTPException(404, f"No OHLC data for {symbol}")

    result = {"symbol": symbol, "candles": candles, "source": "yahoo"}
    _cache_set(cache_key, result)
    print(f"[ohlc] {symbol}: {len(candles)} candles ✓")
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
    "Automobiles": ["TMPV.NS","MARUTI.NS","BAJAJ-AUTO.NS","EICHERMOT.NS","HEROMOTOCO.NS","MOTHERSON.NS","TVSMOTOR.NS","ASHOKLEY.NS"],
    "Energy": ["RELIANCE.NS","ONGC.NS","BPCL.NS","IOC.NS","GAIL.NS","PETRONET.NS","MGL.NS"],
    "Metals": ["TATASTEEL.NS","JSWSTEEL.NS","HINDALCO.NS","COALINDIA.NS","VEDL.NS","NMDC.NS","SAIL.NS","JINDALSTEL.NS"],
    "Infrastructure": ["LT.NS","SIEMENS.NS","ABB.NS","HAVELLS.NS","BHARTIARTL.NS","ADANIPORTS.NS","IRCTC.NS"],
    "Utilities": ["NTPC.NS","POWERGRID.NS","TATAPOWER.NS","TORNTPOWER.NS","ADANIGREEN.NS","CESC.NS"],
    "Telecom": ["BHARTIARTL.NS","IDEA.NS","TATACOMM.NS","HFCL.NS"],
    "Electronics": ["KAYNES.NS","DIXON.NS","AMBER.NS","SYRMA.NS","BEL.NS"],
    "Conglomerate": ["RELIANCE.NS","ADANIENT.NS","ITC.NS","LT.NS","TMPV.NS","M&M.NS"],
    "Real Estate": ["DLF.NS","GODREJPROP.NS","PRESTIGE.NS","PHOENIXLTD.NS","OBEROI.NS","BRIGADE.NS"],
}

# ---------------------------------------------------------------------------
# Industry-level peer map (more specific than sector — fixes broker/bank mixing)
# Yahoo Finance returns an "industry" field inside summaryProfile that is much
# more granular than "sector". For example Angel One's sector="Financial Services"
# but industry="Capital Markets". We check industry FIRST, then fall back to sector.
# ---------------------------------------------------------------------------
_INDUSTRY_PEERS: dict[str, list[str]] = {
    # ── Brokers / Capital Markets ──────────────────────────────────────────
    "Capital Markets":        ["ANGELONE.NS","5PAISA.NS","MOFSL.NS","IIFLSEC.NS","ICICIPRULI.NS","HDFCSEC.NS","ZERODHA.NS"],
    "Investment Banking":     ["ANGELONE.NS","5PAISA.NS","MOFSL.NS","IIFLSEC.NS","ICICIPRULI.NS","HDFCSEC.NS"],
    "Financial Data":         ["ANGELONE.NS","5PAISA.NS","MOFSL.NS","IIFLSEC.NS","CAMS.NS","KFINTECH.NS"],

    # ── Banks ──────────────────────────────────────────────────────────────
    "Banks—Regional":         ["HDFCBANK.NS","ICICIBANK.NS","KOTAKBANK.NS","AXISBANK.NS","SBIN.NS","INDUSINDBK.NS","FEDERALBNK.NS","BANDHANBNK.NS","AUBANK.NS"],
    "Banks—Diversified":      ["HDFCBANK.NS","ICICIBANK.NS","KOTAKBANK.NS","AXISBANK.NS","SBIN.NS","INDUSINDBK.NS","FEDERALBNK.NS","BANDHANBNK.NS","AUBANK.NS"],
    "Banks - Regional":       ["HDFCBANK.NS","ICICIBANK.NS","KOTAKBANK.NS","AXISBANK.NS","SBIN.NS","INDUSINDBK.NS","FEDERALBNK.NS","BANDHANBNK.NS","AUBANK.NS"],
    "Banks - Diversified":    ["HDFCBANK.NS","ICICIBANK.NS","KOTAKBANK.NS","AXISBANK.NS","SBIN.NS","INDUSINDBK.NS","FEDERALBNK.NS","BANDHANBNK.NS","AUBANK.NS"],

    # ── NBFCs / Consumer Finance ───────────────────────────────────────────
    "Consumer Finance":       ["BAJFINANCE.NS","CHOLAFIN.NS","MUTHOOTFIN.NS","MANAPPURAM.NS","MOTHERSON.NS","SHRIRAMFIN.NS","M&MFIN.NS"],
    "Specialty Finance":      ["BAJFINANCE.NS","CHOLAFIN.NS","MUTHOOTFIN.NS","MANAPPURAM.NS","SHRIRAMFIN.NS","M&MFIN.NS"],
    "Credit Services":        ["BAJFINANCE.NS","CHOLAFIN.NS","MUTHOOTFIN.NS","MANAPPURAM.NS","SHRIRAMFIN.NS"],

    # ── Insurance ──────────────────────────────────────────────────────────
    "Insurance—Life":         ["LICI.NS","SBILIFE.NS","HDFCLIFE.NS","MAXLIFE.NS","IPRU.NS","ABSLAMC.NS"],
    "Insurance—Diversified":  ["LICI.NS","SBILIFE.NS","HDFCLIFE.NS","GICRE.NS","NIACL.NS","STARHEALTH.NS"],
    "Insurance - Life":       ["LICI.NS","SBILIFE.NS","HDFCLIFE.NS","MAXLIFE.NS"],
    "Insurance - Diversified":["LICI.NS","SBILIFE.NS","HDFCLIFE.NS","GICRE.NS","NIACL.NS","STARHEALTH.NS"],

    # ── Asset Management ───────────────────────────────────────────────────
    "Asset Management":       ["HDFCAMC.NS","NIPPONLIFE.NS","ABSLAMC.NS","UTIAMC.NS","CAMS.NS","KFINTECH.NS"],

    # ── Software / IT Services ────────────────────────────────────────────
    "Information Technology Services": ["TCS.NS","INFY.NS","WIPRO.NS","HCLTECH.NS","TECHM.NS","LTIM.NS","MPHASIS.NS","PERSISTENT.NS","COFORGE.NS"],
    "Software—Application":   ["TCS.NS","INFY.NS","WIPRO.NS","HCLTECH.NS","TECHM.NS","LTIM.NS","MPHASIS.NS","PERSISTENT.NS","COFORGE.NS"],
    "Software—Infrastructure": ["TCS.NS","INFY.NS","WIPRO.NS","HCLTECH.NS","TECHM.NS","LTIM.NS","MPHASIS.NS"],
    "Software - Application": ["TCS.NS","INFY.NS","WIPRO.NS","HCLTECH.NS","TECHM.NS","LTIM.NS","MPHASIS.NS","PERSISTENT.NS","COFORGE.NS"],
    "IT Services & Consulting": ["TCS.NS","INFY.NS","WIPRO.NS","HCLTECH.NS","TECHM.NS","LTIM.NS","MPHASIS.NS","PERSISTENT.NS","COFORGE.NS"],

    # ── Pharma sub-types ──────────────────────────────────────────────────
    "Drug Manufacturers—General": ["SUNPHARMA.NS","CIPLA.NS","DRREDDY.NS","AUROPHARMA.NS","TORNTPHARM.NS","ALKEM.NS","ZYDUSLIFE.NS"],
    "Drug Manufacturers—Specialty": ["DIVISLAB.NS","DIVI.NS","LAURUS.NS","GRANULES.NS","NATPHARMA.NS"],
    "Pharmaceutical Retailers": ["SUNPHARMA.NS","CIPLA.NS","DRREDDY.NS","AUROPHARMA.NS","TORNTPHARM.NS","ALKEM.NS"],
    "Diagnostics & Research": ["LALPATHLAB.NS","METROPOLIS.NS","THYROCARE.NS","KRSNAA.NS"],

    # ── Auto sub-types ────────────────────────────────────────────────────
    "Auto Manufacturers":     ["TMPV.NS","MARUTI.NS","M&M.NS","BAJAJ-AUTO.NS","EICHERMOT.NS","HEROMOTOCO.NS","TVSMOTOR.NS"],
    "Auto Parts":             ["MOTHERSON.NS","BOSCHLTD.NS","BHARATFORG.NS","APOLLOTYRE.NS","MRF.NS","BALKRISIND.NS"],

    # ── Oil & Gas ─────────────────────────────────────────────────────────
    "Oil & Gas Integrated":   ["RELIANCE.NS","ONGC.NS","BPCL.NS","IOC.NS","GAIL.NS"],
    "Oil & Gas E&P":          ["ONGC.NS","OIL.NS","CAIRNIND.NS"],
    "Oil & Gas Refining":     ["BPCL.NS","IOC.NS","MRPL.NS","HPCL.NS"],

    # ── Steel / Metals ────────────────────────────────────────────────────
    "Steel":                  ["TATASTEEL.NS","JSWSTEEL.NS","SAIL.NS","JINDALSTEL.NS","NMDC.NS"],
    "Aluminum":               ["HINDALCO.NS","NALCO.NS","VEDL.NS"],
    "Other Industrial Metals": ["TATASTEEL.NS","JSWSTEEL.NS","HINDALCO.NS","VEDL.NS","NMDC.NS"],

    # ── Real Estate ───────────────────────────────────────────────────────
    "Real Estate—Development": ["DLF.NS","GODREJPROP.NS","PRESTIGE.NS","PHOENIXLTD.NS","OBEROI.NS","BRIGADE.NS","MAHINDRAP.NS"],
    "Real Estate - Development": ["DLF.NS","GODREJPROP.NS","PRESTIGE.NS","PHOENIXLTD.NS","OBEROI.NS","BRIGADE.NS"],
    "REIT—Diversified":       ["EMBASSY.NS","MINDSPACEREIT.NS","NEXUS.NS"],

    # ── Utilities ─────────────────────────────────────────────────────────
    "Utilities—Regulated Electric": ["NTPC.NS","POWERGRID.NS","TATAPOWER.NS","TORNTPOWER.NS","ADANIGREEN.NS","CESC.NS"],
    "Utilities—Renewable":    ["ADANIGREEN.NS","GREENKO.NS","JSPL.NS","TATAPOWER.NS"],

    # ── Telecom ───────────────────────────────────────────────────────────
    "Telecom Services":       ["BHARTIARTL.NS","IDEA.NS","TATACOMM.NS","HFCL.NS"],

    # ── FMCG / Consumer Staples ───────────────────────────────────────────
    "Household & Personal Products": ["HINDUNILVR.NS","DABUR.NS","MARICO.NS","GODREJCP.NS","COLPAL.NS","EMAMILTD.NS"],
    "Packaged Foods":         ["NESTLEIND.NS","BRITANNIA.NS","TATACONSUM.NS","VARUNBEV.NS","VBL.NS"],

    # ── Retail / Consumer Discretionary ───────────────────────────────────
    "Specialty Retail":       ["TITAN.NS","TRENT.NS","NYKAA.NS","DMART.NS","BATA.NS","RELAXO.NS","VMART.NS"],
    "Department Stores":      ["DMART.NS","TRENT.NS","VMART.NS","SHOPPERSSTOP.NS"],

    # ── Cement ────────────────────────────────────────────────────────────
    "Building Materials":     ["ULTRACEMCO.NS","AMBUJACEM.NS","ACCLTD.NS","SHREECEM.NS","RAMCOCEM.NS","JKCEMENT.NS"],

    # ── Chemicals ─────────────────────────────────────────────────────────
    "Specialty Chemicals":    ["PIDILITIND.NS","AAVAS.NS","NAVINFLUOR.NS","DEEPAKFERT.NS","ATUL.NS","TATACHEM.NS","VINATI.NS"],
    "Agricultural Inputs":    ["PIIND.NS","BAYER.NS","RALLIS.NS","DHANUKA.NS"],

    # ── Capital Goods / Industrials ───────────────────────────────────────
    "Electrical Equipment":   ["SIEMENS.NS","ABB.NS","HAVELLS.NS","POLYCAB.NS","KEI.NS"],
    "Industrial Machinery":   ["LT.NS","BHEL.NS","THERMAX.NS","CUMMINS.NS"],
    "Engineering & Construction": ["LT.NS","NCC.NS","KEC.NS","IRCON.NS","RVNL.NS"],
}

# Symbol-level overrides — verified NSE-listed peers only.
# Groww, Zerodha, Kotak Securities = private/unlisted → excluded.
# ICICIPRULI = insurance (not ICICI Securities which is ISEC).
_SYMBOL_PEER_OVERRIDE: dict[str, list[str]] = {
    # ── Brokers / Stock Exchanges ──────────────────────────────────────────────
    "ANGELONE":   ["ANGELONE.NS","MOFSL.NS","IIFLSEC.NS","5PAISA.NS","ISEC.NS","NUVAMA.NS","EMKAY.NS","GEOJITFSL.NS","SMCGLOBAL.NS","CHOICEIN.NS"],
    "5PAISA":     ["5PAISA.NS","ANGELONE.NS","MOFSL.NS","IIFLSEC.NS","ISEC.NS","NUVAMA.NS","EMKAY.NS","GEOJITFSL.NS"],
    "MOFSL":      ["MOFSL.NS","ANGELONE.NS","IIFLSEC.NS","5PAISA.NS","ISEC.NS","NUVAMA.NS","EMKAY.NS","GEOJITFSL.NS","SMCGLOBAL.NS"],
    "IIFLSEC":    ["IIFLSEC.NS","ANGELONE.NS","MOFSL.NS","5PAISA.NS","ISEC.NS","NUVAMA.NS","EMKAY.NS"],
    "ISEC":       ["ISEC.NS","ANGELONE.NS","MOFSL.NS","IIFLSEC.NS","5PAISA.NS","NUVAMA.NS","EMKAY.NS","GEOJITFSL.NS"],
    "NUVAMA":     ["NUVAMA.NS","MOFSL.NS","IIFLSEC.NS","ISEC.NS","ANGELONE.NS","EMKAY.NS","GEOJITFSL.NS"],
    "EMKAY":      ["EMKAY.NS","MOFSL.NS","IIFLSEC.NS","ISEC.NS","5PAISA.NS","NUVAMA.NS","GEOJITFSL.NS"],
    "GEOJITFSL":  ["GEOJITFSL.NS","ANGELONE.NS","MOFSL.NS","5PAISA.NS","ISEC.NS","EMKAY.NS"],
    "SMCGLOBAL":  ["SMCGLOBAL.NS","ANGELONE.NS","MOFSL.NS","5PAISA.NS","IIFLSEC.NS","CHOICEIN.NS"],
    "CHOICEIN":   ["CHOICEIN.NS","ANGELONE.NS","5PAISA.NS","MOFSL.NS","GEOJITFSL.NS","SMCGLOBAL.NS"],
    "BSELTD":     ["BSELTD.NS","MCX.NS","CDSL.NS","CAMS.NS","KFINTECH.NS","ANGELONE.NS"],
    "MCX":        ["MCX.NS","BSELTD.NS","CDSL.NS","ANGELONE.NS","MOFSL.NS"],
    # ── Market Infrastructure (Depositories / Registrars) ─────────────────────
    "CDSL":       ["CDSL.NS","CAMS.NS","KFINTECH.NS","BSELTD.NS","MCX.NS","ANGELONE.NS","MOFSL.NS"],
    "CAMS":       ["CAMS.NS","KFINTECH.NS","CDSL.NS","BSELTD.NS","HDFCAMC.NS","NIPPONLIFE.NS"],
    "KFINTECH":   ["KFINTECH.NS","CAMS.NS","CDSL.NS","BSELTD.NS","HDFCAMC.NS"],
    # ── Asset Management ──────────────────────────────────────────────────────
    "HDFCAMC":    ["HDFCAMC.NS","NIPPONLIFE.NS","ABSLAMC.NS","UTIAMC.NS","360ONE.NS","MOFSL.NS"],
    "NIPPONLIFE":  ["NIPPONLIFE.NS","HDFCAMC.NS","ABSLAMC.NS","UTIAMC.NS","360ONE.NS"],
    "ABSLAMC":    ["ABSLAMC.NS","HDFCAMC.NS","NIPPONLIFE.NS","UTIAMC.NS","360ONE.NS"],
    "UTIAMC":     ["UTIAMC.NS","HDFCAMC.NS","NIPPONLIFE.NS","ABSLAMC.NS","360ONE.NS"],
    "360ONE":     ["360ONE.NS","HDFCAMC.NS","NIPPONLIFE.NS","ABSLAMC.NS","NUVAMA.NS","MOFSL.NS"],
    # ── Life Insurance ────────────────────────────────────────────────────────
    "SBILIFE":    ["SBILIFE.NS","HDFCLIFE.NS","ICICIPRULI.NS","LICI.NS","MAXLIFE.NS","TATAAIA.NS"],
    "HDFCLIFE":   ["HDFCLIFE.NS","SBILIFE.NS","ICICIPRULI.NS","LICI.NS","MAXLIFE.NS","TATAAIA.NS"],
    "ICICIPRULI": ["ICICIPRULI.NS","SBILIFE.NS","HDFCLIFE.NS","LICI.NS","MAXLIFE.NS"],
    "LICI":       ["LICI.NS","SBILIFE.NS","HDFCLIFE.NS","ICICIPRULI.NS","GICRE.NS","NIACL.NS"],
    "MAXLIFE":    ["MAXLIFE.NS","SBILIFE.NS","HDFCLIFE.NS","ICICIPRULI.NS","LICI.NS"],
    # ── General Insurance ─────────────────────────────────────────────────────
    "STARHEALTH": ["STARHEALTH.NS","NIACL.NS","GICRE.NS","ICICIGI.NS","SBILIFE.NS","HDFCLIFE.NS"],
    "GICRE":      ["GICRE.NS","NIACL.NS","STARHEALTH.NS","ICICIGI.NS"],
    "NIACL":      ["NIACL.NS","GICRE.NS","STARHEALTH.NS","ICICIGI.NS"],
    # ── NBFCs / Consumer Finance ──────────────────────────────────────────────
    "BAJFINANCE": ["BAJFINANCE.NS","CHOLAFIN.NS","SHRIRAMFIN.NS","M&MFIN.NS","MUTHOOTFIN.NS","MANAPPURAM.NS","LTFH.NS","POONAWALLA.NS"],
    "CHOLAFIN":   ["CHOLAFIN.NS","BAJFINANCE.NS","SHRIRAMFIN.NS","M&MFIN.NS","MUTHOOTFIN.NS","LTFH.NS"],
    "MUTHOOTFIN": ["MUTHOOTFIN.NS","MANAPPURAM.NS","CHOLAFIN.NS","BAJFINANCE.NS","SHRIRAMFIN.NS","IIFL.NS"],
    "MANAPPURAM": ["MANAPPURAM.NS","MUTHOOTFIN.NS","CHOLAFIN.NS","BAJFINANCE.NS","IIFL.NS"],
    "SHRIRAMFIN": ["SHRIRAMFIN.NS","BAJFINANCE.NS","CHOLAFIN.NS","M&MFIN.NS","MUTHOOTFIN.NS","LTFH.NS"],
    "M&MFIN":     ["M&MFIN.NS","SHRIRAMFIN.NS","CHOLAFIN.NS","BAJFINANCE.NS","MUTHOOTFIN.NS"],
    "BAJAJFINSV": ["BAJAJFINSV.NS","BAJFINANCE.NS","CHOLAFIN.NS","SHRIRAMFIN.NS","M&MFIN.NS"],
    # ── Large Banks ───────────────────────────────────────────────────────────
    "HDFCBANK":   ["HDFCBANK.NS","ICICIBANK.NS","KOTAKBANK.NS","AXISBANK.NS","SBIN.NS","INDUSINDBK.NS","BANDHANBNK.NS","AUBANK.NS","FEDERALBNK.NS"],
    "ICICIBANK":  ["ICICIBANK.NS","HDFCBANK.NS","KOTAKBANK.NS","AXISBANK.NS","SBIN.NS","INDUSINDBK.NS","BANDHANBNK.NS"],
    "KOTAKBANK":  ["KOTAKBANK.NS","HDFCBANK.NS","ICICIBANK.NS","AXISBANK.NS","SBIN.NS","INDUSINDBK.NS"],
    "AXISBANK":   ["AXISBANK.NS","HDFCBANK.NS","ICICIBANK.NS","KOTAKBANK.NS","SBIN.NS","INDUSINDBK.NS","BANDHANBNK.NS"],
    "SBIN":       ["SBIN.NS","HDFCBANK.NS","ICICIBANK.NS","KOTAKBANK.NS","AXISBANK.NS","INDUSINDBK.NS","BANKBARODA.NS","PNB.NS"],
    "INDUSINDBK": ["INDUSINDBK.NS","AXISBANK.NS","FEDERALBNK.NS","BANDHANBNK.NS","AUBANK.NS","KOTAKBANK.NS"],
    "FEDERALBNK": ["FEDERALBNK.NS","INDUSINDBK.NS","BANDHANBNK.NS","AUBANK.NS","IDFCFIRSTB.NS","KTKBANK.NS"],
    "BANDHANBNK": ["BANDHANBNK.NS","FEDERALBNK.NS","AUBANK.NS","IDFCFIRSTB.NS","INDUSINDBK.NS"],
    "AUBANK":     ["AUBANK.NS","BANDHANBNK.NS","FEDERALBNK.NS","IDFCFIRSTB.NS","INDUSINDBK.NS","UJJIVANSFB.NS"],
    # ── IT Services ───────────────────────────────────────────────────────────
    "TCS":        ["TCS.NS","INFY.NS","WIPRO.NS","HCLTECH.NS","TECHM.NS","LTIM.NS","MPHASIS.NS","PERSISTENT.NS","COFORGE.NS","KPITTECH.NS"],
    "INFY":       ["INFY.NS","TCS.NS","WIPRO.NS","HCLTECH.NS","TECHM.NS","LTIM.NS","MPHASIS.NS","PERSISTENT.NS","COFORGE.NS"],
    "WIPRO":      ["WIPRO.NS","TCS.NS","INFY.NS","HCLTECH.NS","TECHM.NS","LTIM.NS","MPHASIS.NS"],
    "HCLTECH":    ["HCLTECH.NS","TCS.NS","INFY.NS","WIPRO.NS","TECHM.NS","LTIM.NS","MPHASIS.NS","PERSISTENT.NS"],
    "TECHM":      ["TECHM.NS","TCS.NS","INFY.NS","WIPRO.NS","HCLTECH.NS","LTIM.NS","MPHASIS.NS","PERSISTENT.NS","COFORGE.NS"],
    "LTIM":       ["LTIM.NS","TCS.NS","INFY.NS","HCLTECH.NS","TECHM.NS","MPHASIS.NS","PERSISTENT.NS","COFORGE.NS","KPITTECH.NS"],
    "MPHASIS":    ["MPHASIS.NS","LTIM.NS","PERSISTENT.NS","COFORGE.NS","KPITTECH.NS","INFY.NS","WIPRO.NS"],
    "PERSISTENT": ["PERSISTENT.NS","COFORGE.NS","LTIM.NS","MPHASIS.NS","KPITTECH.NS","TECHM.NS"],
    "COFORGE":    ["COFORGE.NS","PERSISTENT.NS","LTIM.NS","MPHASIS.NS","KPITTECH.NS","TECHM.NS"],

    # ── Energy / Oil & Gas ────────────────────────────────────────────────────
    "RELIANCE":   ["ONGC.NS","IOC.NS","BPCL.NS","HPCL.NS","GAIL.NS","BHARTIARTL.NS","ADANIENT.NS","LT.NS","VEDL.NS"],
    "ONGC":       ["ONGC.NS","RELIANCE.NS","IOC.NS","BPCL.NS","HPCL.NS","GAIL.NS","OIL.NS"],
    "IOC":        ["IOC.NS","BPCL.NS","HPCL.NS","ONGC.NS","RELIANCE.NS","GAIL.NS","MRPL.NS"],
    "BPCL":       ["BPCL.NS","IOC.NS","HPCL.NS","ONGC.NS","RELIANCE.NS","MRPL.NS"],
    "HPCL":       ["HPCL.NS","BPCL.NS","IOC.NS","ONGC.NS","RELIANCE.NS","MRPL.NS"],
    "GAIL":       ["GAIL.NS","IGL.NS","MGL.NS","PETRONET.NS","ONGC.NS","RELIANCE.NS"],

    # ── Conglomerates / Infrastructure ───────────────────────────────────────
    "ADANIENT":   ["ADANIENT.NS","RELIANCE.NS","LT.NS","ADANIPORTS.NS","ADANITRANS.NS","VEDL.NS"],
    "LT":         ["LT.NS","SIEMENS.NS","ABB.NS","BHEL.NS","ADANIENT.NS","THERMAX.NS","POWERINDIA.NS"],
    "ITC":        ["ITC.NS","HINDUNILVR.NS","NESTLEIND.NS","BRITANNIA.NS","DABUR.NS","EMAMILTD.NS","GODREJCP.NS"],

    # ── FMCG ──────────────────────────────────────────────────────────────────
    "HINDUNILVR": ["HINDUNILVR.NS","ITC.NS","NESTLEIND.NS","BRITANNIA.NS","DABUR.NS","EMAMILTD.NS","GODREJCP.NS","MARICO.NS"],
    "NESTLEIND":  ["NESTLEIND.NS","HINDUNILVR.NS","BRITANNIA.NS","ITC.NS","DABUR.NS","MARICO.NS","TATACONSUM.NS"],
    "BRITANNIA":  ["BRITANNIA.NS","HINDUNILVR.NS","NESTLEIND.NS","ITC.NS","DABUR.NS","TATACONSUM.NS"],
    "DABUR":      ["DABUR.NS","HINDUNILVR.NS","EMAMILTD.NS","GODREJCP.NS","MARICO.NS","COLPAL.NS"],
    "MARICO":     ["MARICO.NS","DABUR.NS","HINDUNILVR.NS","EMAMILTD.NS","GODREJCP.NS","COLPAL.NS"],
    "COLPAL":     ["COLPAL.NS","DABUR.NS","MARICO.NS","HINDUNILVR.NS","EMAMILTD.NS","GODREJCP.NS"],
    "GODREJCP":   ["GODREJCP.NS","HINDUNILVR.NS","DABUR.NS","MARICO.NS","EMAMILTD.NS","COLPAL.NS"],
    "TATACONSUM": ["TATACONSUM.NS","HINDUNILVR.NS","ITC.NS","NESTLEIND.NS","BRITANNIA.NS","DABUR.NS"],

    # ── Paints ────────────────────────────────────────────────────────────────
    "ASIANPAINT":   ["ASIANPAINT.NS","BERGERPAINTS.NS","KANSAINER.NS","INDIGO.NS","AKZOINDIA.NS"],
    "BERGERPAINTS": ["BERGERPAINTS.NS","ASIANPAINT.NS","KANSAINER.NS","INDIGO.NS","AKZOINDIA.NS"],
    "KANSAINER":    ["KANSAINER.NS","ASIANPAINT.NS","BERGERPAINTS.NS","INDIGO.NS","AKZOINDIA.NS"],

    # ── Metals ────────────────────────────────────────────────────────────────
    "TATASTEEL":  ["TATASTEEL.NS","JSWSTEEL.NS","HINDALCO.NS","VEDL.NS","SAIL.NS","NMDC.NS","JINDALSTEL.NS"],
    "JSWSTEEL":   ["JSWSTEEL.NS","TATASTEEL.NS","HINDALCO.NS","VEDL.NS","SAIL.NS","JINDALSTEL.NS"],
    "HINDALCO":   ["HINDALCO.NS","VEDL.NS","TATASTEEL.NS","JSWSTEEL.NS","NMDC.NS","NALCO.NS"],
    "VEDL":       ["VEDL.NS","HINDALCO.NS","TATASTEEL.NS","JSWSTEEL.NS","NMDC.NS","COALINDIA.NS"],
    "COALINDIA":  ["COALINDIA.NS","NMDC.NS","VEDL.NS","HINDALCO.NS","HINDCOPPER.NS"],
    "SAIL":       ["SAIL.NS","TATASTEEL.NS","JSWSTEEL.NS","JINDALSTEL.NS","RINL.NS","RATNAMANI.NS"],

    # ── Power / Utilities ─────────────────────────────────────────────────────
    "NTPC":       ["NTPC.NS","POWERGRID.NS","TATAPOWER.NS","ADANIGREEN.NS","TORNTPOWER.NS","CESC.NS"],
    "POWERGRID":  ["POWERGRID.NS","NTPC.NS","TATAPOWER.NS","ADANIPOWER.NS","ADANIENT.NS"],
    "TATAPOWER":  ["TATAPOWER.NS","NTPC.NS","POWERGRID.NS","ADANIGREEN.NS","TORNTPOWER.NS","CESC.NS"],
    "ADANIGREEN": ["ADANIGREEN.NS","NTPC.NS","TATAPOWER.NS","SJVN.NS","NHPC.NS","TORNTPOWER.NS"],
    "ADANIPOWER": ["ADANIPOWER.NS","NTPC.NS","TATAPOWER.NS","POWERGRID.NS","CESC.NS"],
    "CESC":       ["CESC.NS","TATAPOWER.NS","NTPC.NS","TORNTPOWER.NS","ADANIPOWER.NS"],

    # ── Pharma ────────────────────────────────────────────────────────────────
    "SUNPHARMA":  ["SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS","AUROPHARMA.NS","LUPIN.NS","ALKEM.NS"],
    "DRREDDY":    ["DRREDDY.NS","SUNPHARMA.NS","CIPLA.NS","DIVISLAB.NS","AUROPHARMA.NS","LUPIN.NS"],
    "CIPLA":      ["CIPLA.NS","SUNPHARMA.NS","DRREDDY.NS","DIVISLAB.NS","LUPIN.NS","AUROPHARMA.NS","ALKEM.NS"],
    "DIVISLAB":   ["DIVISLAB.NS","SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","LUPIN.NS","AUROPHARMA.NS"],
    "LUPIN":      ["LUPIN.NS","SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","AUROPHARMA.NS","ALKEM.NS"],
    "AUROPHARMA": ["AUROPHARMA.NS","SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","LUPIN.NS","ALKEM.NS"],

    # ── Cement ────────────────────────────────────────────────────────────────
    "ULTRACEMCO":  ["ULTRACEMCO.NS","AMBUJACEM.NS","ACC.NS","SHREECEM.NS","RAMCOCEM.NS","JKCEMENT.NS"],
    "AMBUJACEM":   ["AMBUJACEM.NS","ULTRACEMCO.NS","ACC.NS","SHREECEM.NS","RAMCOCEM.NS"],
    "ACC":         ["ACC.NS","AMBUJACEM.NS","ULTRACEMCO.NS","SHREECEM.NS","RAMCOCEM.NS","JKCEMENT.NS"],
    "SHREECEM":    ["SHREECEM.NS","ULTRACEMCO.NS","AMBUJACEM.NS","ACC.NS","RAMCOCEM.NS"],

    # ── Telecom ───────────────────────────────────────────────────────────────
    "BHARTIARTL":  ["BHARTIARTL.NS","RELIANCE.NS","INDUSTOWER.NS","IDEA.NS","TATACOMM.NS","HFCL.NS"],
    "INDUSTOWER":  ["INDUSTOWER.NS","BHARTIARTL.NS","GTPL.NS","TATACOMM.NS"],

    # ── Speciality Chemicals ──────────────────────────────────────────────────
    "SRF":         ["SRF.NS","AAVAS.NS","DEEPAKNITR.NS","NAVINFLUOR.NS","CLEAN.NS","TATACHEM.NS","PCBL.NS"],
    "DEEPAKNITR":  ["DEEPAKNITR.NS","SRF.NS","NAVINFLUOR.NS","TATACHEM.NS","CLEAN.NS","PCBL.NS"],

    # ── Consumer Durables / Electronics ──────────────────────────────────────
    "HAVELLS":    ["HAVELLS.NS","VGUARD.NS","CROMPTON.NS","POLYCAB.NS","KEI.NS","ORIENT.NS"],
    "POLYCAB":    ["POLYCAB.NS","HAVELLS.NS","KEI.NS","VGUARD.NS","CROMPTON.NS"],

    # ── Real Estate ───────────────────────────────────────────────────────────
    "DLF":        ["DLF.NS","GODREJPROP.NS","OBEROIRLTY.NS","PHOENIXLTD.NS","PRESTIGE.NS","BRIGADE.NS"],
    "GODREJPROP": ["GODREJPROP.NS","DLF.NS","OBEROIRLTY.NS","PHOENIXLTD.NS","PRESTIGE.NS"],
}

# ---------------------------------------------------------------------------
# Screener.in peer scraping helpers
# ---------------------------------------------------------------------------
# Screener shows a "Peers" table on every company page — this is more accurate
# than Yahoo Finance industry labels because it uses BSE/NSE sector codes directly.

_SCREENER_SYM_MAP: dict[str, str] = {
    # Screener symbol → Yahoo Finance NS ticker (special cases only)
    "M&M":        "M&M.NS",
    "M&MFIN":     "M&MFIN.NS",
    "BAJAJ-AUTO": "BAJAJ-AUTO.NS",
    "L&TFH":      "L&TFH.NS",
    "L&T":        "LT.NS",
}


def _screener_sym_to_ns(sym: str) -> str:
    """Convert a Screener.in symbol to a Yahoo Finance .NS ticker."""
    return _SCREENER_SYM_MAP.get(sym, f"{sym}.NS")


def _parse_screener_peers(soup, self_symbol: str) -> list[str]:
    """Extract peer symbols (as .NS tickers) from Screener peers section."""
    if not soup or not _BS4_AVAILABLE:
        return []
    peers, seen = [], {self_symbol.upper()}
    try:
        sec = soup.find(id="peers") or soup.find("section", id="peers") or soup.find("div", id="peers")
        if not sec:
            return []
        for link in sec.find_all("a", href=True):
            href = link.get("href", "")
            if "/company/" not in href:
                continue
            parts = [p for p in href.strip("/").split("/") if p]
            if not parts or parts[0] != "company":
                continue
            sym = parts[1].upper() if len(parts) > 1 else ""
            if not sym or sym in seen:
                continue
            seen.add(sym)
            peers.append(_screener_sym_to_ns(sym))
            if len(peers) >= 12:
                break
    except Exception as e:
        print(f"[Screener peers] parse error for {self_symbol}: {e}")
    return peers


def _parse_screener_peers_with_metrics(soup, self_symbol: str) -> list[dict]:
    """
    Parse Screener's peers table INCLUDING financial metrics.
    Returns list of peer dicts with symbol, name, price, PE, market cap, ROE, etc.
    This is 100% Screener-sourced — no Yahoo Finance needed.
    """
    if not soup or not _BS4_AVAILABLE:
        return []

    peers_section = (
        soup.find(id="peers")
        or soup.find("section", id="peers")
        or soup.find("div", id="peers")
    )
    if not peers_section:
        return []

    table = peers_section.find("table")
    if not table:
        return []

    # ── Parse column headers ──────────────────────────────────────────────
    headers = []
    thead = table.find("thead")
    if thead:
        for th in thead.find_all("th"):
            headers.append(th.get_text(strip=True).lower())

    def col_idx(*keywords):
        for i, h in enumerate(headers):
            for kw in keywords:
                if kw in h:
                    return i
        return -1

    name_idx   = 0
    price_idx  = col_idx("cmp", "price", "ltp")
    pe_idx     = col_idx("p/e", " pe ")
    mktcap_idx = col_idx("mar cap", "market cap", "mcap")
    roe_idx    = col_idx("roe")
    roce_idx   = col_idx("roce")
    div_idx    = col_idx("div yld", "dividend yield")
    netpat_idx = col_idx("np qtr", "net profit", "pat qtr")

    # ── Parse rows ────────────────────────────────────────────────────────
    result = []
    seen   = set()
    self_sym_clean = self_symbol.upper().replace(".NS", "").replace(".BO", "")

    tbody = table.find("tbody")
    if not tbody:
        return []

    for tr in tbody.find_all("tr"):
        cells = tr.find_all("td")
        if not cells:
            continue
        try:
            # Symbol from link in first cell
            link = cells[0].find("a", href=True)
            if not link:
                continue
            href  = link.get("href", "")
            parts = [p for p in href.strip("/").split("/") if p]
            if not parts or parts[0] != "company":
                continue
            sym  = parts[1].upper() if len(parts) > 1 else ""
            if not sym or sym in seen:
                continue
            seen.add(sym)

            name    = link.get_text(strip=True)
            is_self = sym == self_sym_clean

            def _cell(idx):
                if idx < 0 or idx >= len(cells):
                    return None
                txt = cells[idx].get_text(strip=True).replace(",", "").replace("%", "").strip()
                if txt in ("", "-", "--", "N/A"):
                    return None
                try:
                    return float(txt)
                except (ValueError, TypeError):
                    return None

            price_v  = _cell(price_idx)
            pe_v     = _cell(pe_idx)
            mktcap_v = _cell(mktcap_idx)   # ₹ Cr
            roe_v    = _cell(roe_idx)
            roce_v   = _cell(roce_idx)
            div_v    = _cell(div_idx)

            # Sanity caps
            if pe_v is not None and (pe_v < 0 or pe_v > 500):
                pe_v = None

            # Derive P/B from price / book value (not directly in table — computed later)
            result.append({
                "symbol":       sym,
                "name":         name,
                "marketCap":    round(mktcap_v, 0) if mktcap_v else None,
                "currentPrice": round(price_v, 2)  if price_v  else None,
                "pe":           round(pe_v, 1)      if pe_v     else None,
                "pb":           None,               # not in Screener peers table
                "evEbitda":     None,               # not in Screener peers table
                "revenueGrowth":None,
                "netMargin":    None,
                "roe":          round(roe_v or roce_v, 1) if (roe_v or roce_v) else None,
                "de":           None,               # not in Screener peers table
                "isSelf":       is_self,
            })
        except Exception:
            continue

    return result


@app.get("/peers/{symbol}")
def get_peers(symbol: str):
    """
    Peer comparison — 100% Screener.in. No Yahoo Finance.

    Priority:
      1. Symbol override (verified correct peers for known stocks)
      2. Screener peers table WITH metrics (fastest, most accurate)
      3. Sector/industry fallback from curated maps (if Screener unavailable)
    """
    symbol    = symbol.upper().strip()
    cache_key = f"peers:{symbol}"
    cached    = _cache_get(cache_key)
    if cached is not None:
        return cached

    bare = symbol.replace(".NS","").replace(".BO","")

    # ── 1. Fetch Screener page (cached from company-v2 load if already done) ─
    soup = _fetch_screener_page(symbol)

    # ── 2. Get sector/industry from Screener ratios ───────────────────────────
    sector   = ""
    industry = ""
    if soup:
        ratios   = _parse_screener_ratios(soup)
        sector   = ratios.get("screenerIndustry") or ratios.get("sector") or ""
        industry = ratios.get("screenerIndustry") or ""
    if not sector:
        sector = (STOCK_UNIVERSE.get(symbol) or {}).get("sector", "")

    # ── 3. Build peer list ────────────────────────────────────────────────────
    results: list[dict] = []

    # 3a. Symbol override always wins (verified correct peers)
    if bare in _SYMBOL_PEER_OVERRIDE:
        peer_syms = [p.replace(".NS","").replace(".BO","")
                     for p in _SYMBOL_PEER_OVERRIDE[bare]
                     if p.replace(".NS","").replace(".BO","").upper() != bare][:7]
        print(f"[peers] {symbol}: symbol override → {len(peer_syms)} peers")

        # Build self row from cached company-v2
        self_cv2 = _cache_get(f"company_v2:{symbol}")
        if self_cv2:
            results.append({
                "symbol": bare,
                "name": self_cv2.get("name", bare),
                "marketCap": self_cv2.get("marketCap"),
                "currentPrice": self_cv2.get("currentPrice"),
                "pe": self_cv2.get("pe"),
                "pb": self_cv2.get("pb"),
                "evEbitda": None, "revenueGrowth": None, "netMargin": None,
                "roe": self_cv2.get("roe"),
                "de": self_cv2.get("debtToEquity"),
                "isSelf": True,
            })

        # Fetch peer company-v2 data (fast — hits same Screener page)
        for peer_sym in peer_syms:
            try:
                peer_cv2 = _cache_get(f"company_v2:{peer_sym}")
                if not peer_cv2:
                    peer_soup = _fetch_screener_page(peer_sym)
                    if peer_soup:
                        pr = _parse_screener_ratios(peer_soup)
                        h1 = peer_soup.find("h1")
                        pname = h1.get_text(strip=True) if h1 else peer_sym
                        peer_cv2 = {
                            "name": pname, "pe": pr.get("pe"), "pb": pr.get("pb"),
                            "roe": pr.get("roe"), "marketCap": pr.get("marketCap"),
                            "currentPrice": pr.get("currentPrice"),
                            "debtToEquity": pr.get("debtToEquity"),
                        }
                        _cache_set(f"company_v2:{peer_sym}", peer_cv2)
                if peer_cv2:
                    results.append({
                        "symbol": peer_sym,
                        "name": peer_cv2.get("name", peer_sym),
                        "marketCap": peer_cv2.get("marketCap"),
                        "currentPrice": peer_cv2.get("currentPrice"),
                        "pe": peer_cv2.get("pe"),
                        "pb": peer_cv2.get("pb"),
                        "evEbitda": None, "revenueGrowth": None, "netMargin": None,
                        "roe": peer_cv2.get("roe"),
                        "de": peer_cv2.get("debtToEquity"),
                        "isSelf": False,
                    })
            except Exception:
                pass

    # 3b. Screener peers table WITH metrics (live — no symbol override available)
    elif soup:
        screener_results = _parse_screener_peers_with_metrics(soup, symbol)
        if screener_results:
            # Mark self row
            for row in screener_results:
                if row["symbol"] == bare:
                    row["isSelf"] = True
            # If self not in screener table, prepend from company-v2
            has_self = any(r["isSelf"] for r in screener_results)
            if not has_self:
                self_cv2 = _cache_get(f"company_v2:{symbol}")
                if self_cv2:
                    screener_results.insert(0, {
                        "symbol": bare, "name": self_cv2.get("name", bare),
                        "marketCap": self_cv2.get("marketCap"),
                        "currentPrice": self_cv2.get("currentPrice"),
                        "pe": self_cv2.get("pe"), "pb": self_cv2.get("pb"),
                        "evEbitda": None, "revenueGrowth": None, "netMargin": None,
                        "roe": self_cv2.get("roe"), "de": self_cv2.get("debtToEquity"),
                        "isSelf": True,
                    })
            results = screener_results[:8]
            print(f"[peers] {symbol}: Screener table → {len(results)} peers ✓")

    # 3c. Curated fallback (Screener unavailable)
    if not results:
        peer_ns_list = (
            _INDUSTRY_PEERS.get(industry, [])
            or _SECTOR_PEERS.get(sector, [])
        )
        peer_syms = [p.replace(".NS","").replace(".BO","")
                     for p in peer_ns_list
                     if p.replace(".NS","").replace(".BO","").upper() != bare][:6]
        for peer_sym in [bare] + peer_syms:
            cv2 = _cache_get(f"company_v2:{peer_sym}")
            if cv2:
                results.append({
                    "symbol": peer_sym, "name": cv2.get("name", peer_sym),
                    "marketCap": cv2.get("marketCap"), "currentPrice": cv2.get("currentPrice"),
                    "pe": cv2.get("pe"), "pb": cv2.get("pb"),
                    "evEbitda": None, "revenueGrowth": None, "netMargin": None,
                    "roe": cv2.get("roe"), "de": cv2.get("debtToEquity"),
                    "isSelf": peer_sym == bare,
                })
        print(f"[peers] {symbol}: curated fallback → {len(results)} peers")

    result = {"sector": industry or sector, "industry": industry, "peers": results, "source": "screener"}
    _cache_set(cache_key, result)
    return result


@app.get("/universe/size")
def universe_size():
    return {"count": len(STOCK_UNIVERSE), "source": "NSE EQUITY_L.csv"}


# ── Quarterly Results endpoint ────────────────────────────────────────────────
@app.get("/quarterly/{symbol}")
def quarterly_results(symbol: str):
    """
    Quarterly P&L from Screener.in (#quarters table).
    Returns last 8 quarters (newest first) with revenue, PAT, EPS, OPM.
    """
    symbol = symbol.upper().strip()
    cache_key = f"quarterly:{symbol}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    soup = _fetch_screener_page(symbol)
    if not soup:
        raise HTTPException(404, f"Company not found on Screener: {symbol}")

    section = soup.find(id="quarters")
    if not section:
        raise HTTPException(404, f"Quarterly data not found for {symbol}")

    table = section.find("table")
    if not table:
        raise HTTPException(404, f"Quarterly table missing for {symbol}")

    # Parse headers → quarter labels like "Jun 2024", "Sep 2024"
    quarters = []
    thead = table.find("thead")
    if thead:
        for th in thead.find_all("th")[1:]:
            txt = th.get_text(strip=True)
            if txt and txt.lower() != "ttm":
                quarters.append(txt)

    # Parse rows
    rows: dict = {}
    tbody = table.find("tbody")
    if tbody:
        for tr in tbody.find_all("tr"):
            cells = tr.find_all("td")
            if not cells:
                continue
            label = cells[0].get_text(strip=True).lower().rstrip(" +").strip()
            vals = []
            for td in cells[1: len(quarters) + 1]:
                txt = td.get_text(strip=True)
                vals.append(_cr(txt))
            rows[label] = vals

    def _row(*keywords):
        for kw in keywords:
            for label, vals in rows.items():
                if label == kw:
                    return vals
        for kw in keywords:
            for label, vals in rows.items():
                if label.startswith(kw):
                    return vals
        for kw in keywords:
            for label, vals in rows.items():
                if kw in label:
                    return vals
        return [0.0] * len(quarters)

    revenues  = _row("sales", "revenue", "net interest income", "total income")
    pats      = _row("net profit", "profit after tax", "pat")
    opm_vals  = _row("opm", "operating profit margin", "ebitda margin")
    eps_vals  = _row("eps")

    # Build result list — newest quarter first
    results = []
    for i, q in enumerate(quarters):
        rev  = revenues[i]  if i < len(revenues)  else 0.0
        pat  = pats[i]      if i < len(pats)      else 0.0
        opm  = opm_vals[i]  if i < len(opm_vals)  else 0.0
        eps  = eps_vals[i]  if i < len(eps_vals)  else 0.0

        results.append({
            "quarter": q,          # e.g. "Jun 2024"
            "revenue": rev,        # ₹ Crore
            "pat":     pat,        # ₹ Crore
            "opm":     opm,        # %
            "eps":     eps,        # ₹
        })

    # Newest first
    results.reverse()

    _cache_set(cache_key, results)
    return results


# ── Stock Screener endpoint ───────────────────────────────────────────────────
@app.get("/screener")
def stock_screener(
    min_roe: float = 0,
    max_pe: float = 9999,
    min_rev_growth: float = -999,
    min_net_margin: float = -999,
    max_debt_equity: float = 9999,
    sector: str = "",
    limit: int = 30,
):
    """
    Filter stocks from the NSE universe using cached Screener.in data.
    Scores 0-100 per stock. Returns top `limit` matches.

    Query params:
      min_roe          — minimum ROE %          (default: 0)
      max_pe           — maximum P/E ratio       (default: no limit)
      min_rev_growth   — minimum revenue CAGR %  (default: no limit)
      min_net_margin   — minimum net margin %    (default: no limit)
      max_debt_equity  — maximum D/E ratio        (default: no limit)
      sector           — sector filter string     (default: all)
      limit            — max results returned     (default: 30)
    """

    # We'll screen from the universe list — fetch company-v2 for cached symbols
    # Use the in-memory cache to avoid re-scraping

    results = []

    # Try to find pre-cached company-v2 data
    cached_symbols = [k.replace("company_v2:", "") for k in _cache.keys() if k.startswith("company_v2:")]

    for sym in cached_symbols[:200]:  # cap at 200 to avoid timeout
        try:
            cached_entry = _cache.get(f"company_v2:{sym}")
            data = cached_entry[0] if cached_entry else None
            if not data:
                continue

            roe      = float(data.get("roe", 0) or 0)
            pe       = float(data.get("pe", 9999) or 9999)
            de       = float(data.get("debtToEquity", 0) or 0)
            margin   = float(data.get("netMargin", 0) or 0)
            stk_sect = str(data.get("sector", "") or "")
            name     = str(data.get("name", sym))
            price    = float(data.get("currentPrice", 0) or 0)
            mktcap   = float(data.get("marketCap", 0) or 0)

            # Apply filters
            if roe < min_roe:
                continue
            if pe > max_pe and pe > 0:
                continue
            if de > max_debt_equity:
                continue
            if margin < min_net_margin:
                continue
            if sector and sector.lower() not in stk_sect.lower():
                continue

            # Simple quality score
            score = 0
            if roe >= 20:   score += 30
            elif roe >= 15: score += 20
            elif roe >= 10: score += 10
            if pe > 0 and pe <= 20: score += 25
            elif pe <= 35:          score += 15
            if de < 0.5:  score += 20
            elif de < 1:  score += 10
            if margin >= 15: score += 25
            elif margin >= 8: score += 15

            results.append({
                "symbol":    sym,
                "name":      name,
                "sector":    stk_sect,
                "price":     price,
                "marketCap": mktcap,
                "pe":        pe if pe < 9999 else None,
                "roe":       roe,
                "debtToEquity": de,
                "netMargin": margin,
                "score":     score,
            })
        except Exception:
            continue

    # Sort by quality score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:limit]


# ---------------------------------------------------------------------------
# /screener-debug  — returns raw Screener.in CSV so we can inspect headers
# ---------------------------------------------------------------------------
@app.get("/screener-debug")
def screener_debug():
    """Hit Screener.in with a minimal query and return the raw CSV text + headers."""
    import urllib.parse as _up
    fields_encoded = "Name,Current+Price,Market+Capitalization,Price+to+Earning,Return+on+equity,Net+profit+margin"
    url = f"https://www.screener.in/screen/raw/?sort=Market+Capitalization&query=Return+on+equity+%3E+15&fields={fields_encoded}"
    try:
        sess = _get_screener_session()
        if not sess:
            return {"error": "No session", "url": url}
        csv_headers = {**_make_browser_headers("https://www.screener.in/screens/"), "Accept": "text/csv,text/plain,*/*"}
        resp = sess.get(url, headers=csv_headers, timeout=20)
        raw = resp.text[:2000]  # first 2000 chars
        lines = raw.split("\n")
        return {
            "status":      resp.status_code,
            "url":         url,
            "first_line":  lines[0] if lines else "",
            "second_line": lines[1] if len(lines) > 1 else "",
            "raw_preview": raw,
        }
    except Exception as e:
        return {"error": str(e)}

# ---------------------------------------------------------------------------
# /screener-v2  — Twelve Data API (reliable, works on Railway, free tier)
# ---------------------------------------------------------------------------
# Architecture:
#   Seed: 50 hardcoded top NSE stocks → screener works INSTANTLY on first open
#   Twelve Data: updates all 500+ NSE stocks daily via /statistics endpoint
#   Cache: refreshed once daily (stays within free 800 credits/day limit)
#   Query: filters in-memory cache → instant results
#
# Setup: Set TWELVE_DATA_API_KEY env var on Railway
# Free tier: 800 credits/day = 400 stocks/day (refreshes top 400 daily)
# ---------------------------------------------------------------------------
import threading as _threading
import hashlib as _hashlib

TWELVE_DATA_KEY = os.environ.get("TWELVE_DATA_API_KEY", "")

# ── Hardcoded seed — top 50 NSE stocks (always available, zero API calls) ────
_SEED = [
    {"symbol":"RELIANCE",  "name":"Reliance Industries",       "sector":"Energy",             "price":2950, "marketCap":1998000,"pe":27.5,"roe":14.2,"roce":12.8,"netMargin":8.4, "debtToEquity":0.43,"revenueGrowth5Y":12.3,"dividendYield":0.3},
    {"symbol":"TCS",       "name":"Tata Consultancy Services", "sector":"Technology",         "price":4120, "marketCap":1492000,"pe":30.2,"roe":52.8,"roce":68.1,"netMargin":19.2,"debtToEquity":0.0, "revenueGrowth5Y":11.8,"dividendYield":1.7},
    {"symbol":"HDFCBANK",  "name":"HDFC Bank",                 "sector":"Financial Services", "price":1920, "marketCap":1462000,"pe":19.8,"roe":16.5,"roce":0.0, "netMargin":24.3,"debtToEquity":7.2, "revenueGrowth5Y":18.6,"dividendYield":1.1},
    {"symbol":"BHARTIARTL","name":"Bharti Airtel",             "sector":"Communication",      "price":1920, "marketCap":1139000,"pe":82.1,"roe":18.4,"roce":12.3,"netMargin":11.2,"debtToEquity":1.8, "revenueGrowth5Y":14.2,"dividendYield":0.4},
    {"symbol":"ICICIBANK", "name":"ICICI Bank",                "sector":"Financial Services", "price":1430, "marketCap":1009000,"pe":18.6,"roe":17.8,"roce":0.0, "netMargin":28.4,"debtToEquity":6.8, "revenueGrowth5Y":20.1,"dividendYield":0.7},
    {"symbol":"INFOSYS",   "name":"Infosys",                   "sector":"Technology",         "price":1920, "marketCap":800000, "pe":25.4,"roe":32.4,"roce":41.2,"netMargin":16.8,"debtToEquity":0.07,"revenueGrowth5Y":10.2,"dividendYield":2.8},
    {"symbol":"SBIN",      "name":"State Bank of India",       "sector":"Financial Services", "price":830,  "marketCap":740000, "pe":10.2,"roe":18.2,"roce":0.0, "netMargin":22.1,"debtToEquity":12.1,"revenueGrowth5Y":14.8,"dividendYield":1.8},
    {"symbol":"HINDUNILVR","name":"Hindustan Unilever",        "sector":"Consumer Goods",     "price":2680, "marketCap":628000, "pe":54.2,"roe":20.1,"roce":26.8,"netMargin":16.2,"debtToEquity":0.0, "revenueGrowth5Y":8.4, "dividendYield":1.5},
    {"symbol":"ITC",       "name":"ITC",                       "sector":"Consumer Goods",     "price":470,  "marketCap":589000, "pe":28.4,"roe":27.8,"roce":34.1,"netMargin":28.6,"debtToEquity":0.0, "revenueGrowth5Y":9.8, "dividendYield":3.1},
    {"symbol":"BAJFINANCE","name":"Bajaj Finance",             "sector":"Financial Services", "price":7800, "marketCap":470000, "pe":31.2,"roe":21.4,"roce":0.0, "netMargin":24.8,"debtToEquity":3.8, "revenueGrowth5Y":28.4,"dividendYield":0.3},
    {"symbol":"LT",        "name":"Larsen & Toubro",           "sector":"Capital Goods",      "price":3680, "marketCap":520000, "pe":32.1,"roe":14.2,"roce":16.8,"netMargin":7.8, "debtToEquity":0.8, "revenueGrowth5Y":12.1,"dividendYield":0.8},
    {"symbol":"KOTAKBANK", "name":"Kotak Mahindra Bank",       "sector":"Financial Services", "price":1980, "marketCap":394000, "pe":20.8,"roe":14.1,"roce":0.0, "netMargin":26.4,"debtToEquity":5.2, "revenueGrowth5Y":16.8,"dividendYield":0.1},
    {"symbol":"HCLTECH",   "name":"HCL Technologies",          "sector":"Technology",         "price":1920, "marketCap":521000, "pe":28.4,"roe":24.8,"roce":31.2,"netMargin":14.2,"debtToEquity":0.11,"revenueGrowth5Y":13.4,"dividendYield":3.4},
    {"symbol":"MARUTI",    "name":"Maruti Suzuki India",       "sector":"Automobile",         "price":12800,"marketCap":398000, "pe":26.4,"roe":14.8,"roce":18.4,"netMargin":7.2, "debtToEquity":0.01,"revenueGrowth5Y":11.8,"dividendYield":0.7},
    {"symbol":"ASIANPAINT","name":"Asian Paints",              "sector":"Consumer Goods",     "price":2850, "marketCap":272000, "pe":52.8,"roe":30.2,"roce":38.4,"netMargin":14.8,"debtToEquity":0.02,"revenueGrowth5Y":11.2,"dividendYield":1.1},
    {"symbol":"AXISBANK",  "name":"Axis Bank",                 "sector":"Financial Services", "price":1240, "marketCap":383000, "pe":14.8,"roe":17.2,"roce":0.0, "netMargin":22.8,"debtToEquity":8.4, "revenueGrowth5Y":18.2,"dividendYield":0.1},
    {"symbol":"TITAN",     "name":"Titan Company",             "sector":"Consumer Goods",     "price":3650, "marketCap":324000, "pe":86.2,"roe":28.4,"roce":34.1,"netMargin":7.8, "debtToEquity":0.0, "revenueGrowth5Y":21.4,"dividendYield":0.4},
    {"symbol":"SUNPHARMA", "name":"Sun Pharmaceutical",        "sector":"Healthcare",         "price":1920, "marketCap":461000, "pe":38.4,"roe":14.8,"roce":16.2,"netMargin":19.2,"debtToEquity":0.08,"revenueGrowth5Y":9.8, "dividendYield":0.9},
    {"symbol":"WIPRO",     "name":"Wipro",                     "sector":"Technology",         "price":570,  "marketCap":298000, "pe":24.2,"roe":16.8,"roce":21.4,"netMargin":14.4,"debtToEquity":0.15,"revenueGrowth5Y":7.8, "dividendYield":0.2},
    {"symbol":"NTPC",      "name":"NTPC",                      "sector":"Power",              "price":380,  "marketCap":369000, "pe":18.4,"roe":12.8,"roce":10.4,"netMargin":18.2,"debtToEquity":1.2, "revenueGrowth5Y":8.4, "dividendYield":2.1},
    {"symbol":"TMPV","name":"Tata Motors PV",                  "sector":"Automobile",         "price":398,  "marketCap":143000, "pe":10.2,"roe":31.4,"roce":18.8,"netMargin":5.8, "debtToEquity":1.4, "revenueGrowth5Y":18.4,"dividendYield":0.5},
    {"symbol":"TECHM",     "name":"Tech Mahindra",             "sector":"Technology",         "price":1720, "marketCap":167000, "pe":38.2,"roe":14.8,"roce":18.2,"netMargin":8.4, "debtToEquity":0.08,"revenueGrowth5Y":8.8, "dividendYield":1.4},
    {"symbol":"COALINDIA", "name":"Coal India",                "sector":"Metals & Mining",    "price":480,  "marketCap":295000, "pe":8.4, "roe":42.8,"roce":52.1,"netMargin":18.4,"debtToEquity":0.0, "revenueGrowth5Y":8.2, "dividendYield":5.8},
    {"symbol":"DIVISLAB",  "name":"Divi's Laboratories",       "sector":"Healthcare",         "price":5800, "marketCap":154000, "pe":68.4,"roe":18.4,"roce":21.8,"netMargin":24.8,"debtToEquity":0.0, "revenueGrowth5Y":11.4,"dividendYield":0.7},
    {"symbol":"CIPLA",     "name":"Cipla",                     "sector":"Healthcare",         "price":1620, "marketCap":131000, "pe":28.4,"roe":14.2,"roce":17.8,"netMargin":14.8,"debtToEquity":0.04,"revenueGrowth5Y":12.8,"dividendYield":0.5},
    {"symbol":"DRREDDY",   "name":"Dr. Reddy's Laboratories",  "sector":"Healthcare",         "price":6800, "marketCap":113000, "pe":22.4,"roe":18.4,"roce":22.8,"netMargin":16.8,"debtToEquity":0.12,"revenueGrowth5Y":14.2,"dividendYield":0.6},
    {"symbol":"ONGC",      "name":"ONGC",                      "sector":"Oil & Gas",          "price":310,  "marketCap":391000, "pe":7.8, "roe":11.4,"roce":12.8,"netMargin":12.4,"debtToEquity":0.3, "revenueGrowth5Y":8.8, "dividendYield":4.2},
    {"symbol":"EICHERMOT", "name":"Eicher Motors",             "sector":"Automobile",         "price":5200, "marketCap":143000, "pe":32.4,"roe":24.8,"roce":30.2,"netMargin":20.8,"debtToEquity":0.0, "revenueGrowth5Y":12.4,"dividendYield":1.2},
    {"symbol":"BRITANNIA", "name":"Britannia Industries",      "sector":"Consumer Goods",     "price":5600, "marketCap":135000, "pe":54.2,"roe":42.8,"roce":56.4,"netMargin":12.4,"debtToEquity":0.28,"revenueGrowth5Y":9.8, "dividendYield":1.4},
    {"symbol":"JSWSTEEL",  "name":"JSW Steel",                 "sector":"Metals & Mining",    "price":980,  "marketCap":240000, "pe":18.4,"roe":14.2,"roce":14.8,"netMargin":5.8, "debtToEquity":1.2, "revenueGrowth5Y":14.8,"dividendYield":0.8},
    {"symbol":"TATASTEEL", "name":"Tata Steel",                "sector":"Metals & Mining",    "price":168,  "marketCap":210000, "pe":18.8,"roe":8.4, "roce":10.2,"netMargin":4.2, "debtToEquity":1.6, "revenueGrowth5Y":14.2,"dividendYield":1.2},
    {"symbol":"HINDALCO",  "name":"Hindalco Industries",       "sector":"Metals & Mining",    "price":680,  "marketCap":152000, "pe":14.2,"roe":12.8,"roce":11.4,"netMargin":5.8, "debtToEquity":0.9, "revenueGrowth5Y":18.4,"dividendYield":0.7},
    {"symbol":"ADANIPORTS","name":"Adani Ports & SEZ",         "sector":"Services",           "price":1380, "marketCap":298000, "pe":28.4,"roe":14.8,"roce":12.4,"netMargin":28.4,"debtToEquity":1.4, "revenueGrowth5Y":18.2,"dividendYield":0.5},
    {"symbol":"NESTLEIND", "name":"Nestle India",              "sector":"Consumer Goods",     "price":2420, "marketCap":233400, "pe":72.4,"roe":118.4,"roce":152.1,"netMargin":14.8,"debtToEquity":0.0,"revenueGrowth5Y":10.4,"dividendYield":2.8},
    {"symbol":"PIDILITIND","name":"Pidilite Industries",       "sector":"Chemicals",          "price":2950, "marketCap":150000, "pe":78.4,"roe":24.8,"roce":31.4,"netMargin":14.4,"debtToEquity":0.08,"revenueGrowth5Y":14.2,"dividendYield":0.5},
    {"symbol":"PAGEIND",   "name":"Page Industries",           "sector":"Textiles",           "price":42800,"marketCap":47800,  "pe":68.4,"roe":58.4,"roce":72.1,"netMargin":13.4,"debtToEquity":0.0, "revenueGrowth5Y":12.8,"dividendYield":1.0},
    {"symbol":"COLPAL",    "name":"Colgate-Palmolive India",   "sector":"Consumer Goods",     "price":2980, "marketCap":81000,  "pe":52.8,"roe":68.4,"roce":87.2,"netMargin":16.4,"debtToEquity":0.0, "revenueGrowth5Y":7.4, "dividendYield":1.5},
    {"symbol":"ZOMATO",    "name":"Zomato",                    "sector":"Consumer Services",  "price":268,  "marketCap":237400, "pe":None,"roe":4.8, "roce":3.4, "netMargin":4.2, "debtToEquity":0.0, "revenueGrowth5Y":82.4,"dividendYield":0.0},
    {"symbol":"IRCTC",     "name":"IRCTC",                     "sector":"Services",           "price":840,  "marketCap":67400,  "pe":58.4,"roe":28.4,"roce":34.8,"netMargin":28.4,"debtToEquity":0.0, "revenueGrowth5Y":28.4,"dividendYield":0.8},
    {"symbol":"PERSISTENT","name":"Persistent Systems",        "sector":"Technology",         "price":5200, "marketCap":80200,  "pe":52.4,"roe":24.8,"roce":31.4,"netMargin":14.8,"debtToEquity":0.0, "revenueGrowth5Y":28.4,"dividendYield":0.8},
    {"symbol":"TATAELXSI", "name":"Tata Elxsi",                "sector":"Technology",         "price":7800, "marketCap":48600,  "pe":48.4,"roe":38.4,"roce":48.2,"netMargin":22.8,"debtToEquity":0.0, "revenueGrowth5Y":28.8,"dividendYield":1.4},
    {"symbol":"INDIGO",    "name":"IndiGo (InterGlobe)",       "sector":"Services",           "price":4200, "marketCap":162000, "pe":18.4,"roe":158.4,"roce":28.4,"netMargin":8.4,"debtToEquity":2.8,"revenueGrowth5Y":24.8,"dividendYield":0.5},
    {"symbol":"NAUKRI",    "name":"Info Edge India",           "sector":"Technology",         "price":8200, "marketCap":70200,  "pe":98.4,"roe":14.4,"roce":16.8,"netMargin":28.4,"debtToEquity":0.0, "revenueGrowth5Y":18.4,"dividendYield":0.1},
    {"symbol":"BAJAJ-AUTO","name":"Bajaj Auto",                "sector":"Automobile",         "price":9200, "marketCap":258000, "pe":28.4,"roe":24.8,"roce":30.2,"netMargin":16.8,"debtToEquity":0.0, "revenueGrowth5Y":12.4,"dividendYield":1.8},
    {"symbol":"MPHASIS",   "name":"Mphasis",                   "sector":"Technology",         "price":2980, "marketCap":55800,  "pe":28.4,"roe":22.4,"roce":28.4,"netMargin":14.4,"debtToEquity":0.0, "revenueGrowth5Y":18.4,"dividendYield":1.8},
    {"symbol":"COFORGE",   "name":"Coforge",                   "sector":"Technology",         "price":7200, "marketCap":44800,  "pe":48.4,"roe":28.4,"roce":34.8,"netMargin":8.8, "debtToEquity":0.4, "revenueGrowth5Y":24.8,"dividendYield":0.8},
    {"symbol":"DABUR",     "name":"Dabur India",               "sector":"Consumer Goods",     "price":558,  "marketCap":98900,  "pe":52.4,"roe":18.4,"roce":22.8,"netMargin":14.2,"debtToEquity":0.0, "revenueGrowth5Y":8.8, "dividendYield":1.1},
    {"symbol":"MARICO",    "name":"Marico",                    "sector":"Consumer Goods",     "price":680,  "marketCap":88000,  "pe":48.2,"roe":38.4,"roce":48.2,"netMargin":14.8,"debtToEquity":0.0, "revenueGrowth5Y":7.8, "dividendYield":1.8},
    {"symbol":"HAVELLS",   "name":"Havells India",             "sector":"Consumer Goods",     "price":1820, "marketCap":113900, "pe":64.8,"roe":22.8,"roce":28.4,"netMargin":8.4, "debtToEquity":0.02,"revenueGrowth5Y":14.8,"dividendYield":0.6},
    {"symbol":"MUTHOOTFIN","name":"Muthoot Finance",           "sector":"Financial Services", "price":1920, "marketCap":77400,  "pe":18.4,"roe":18.4,"roce":0.0, "netMargin":28.4,"debtToEquity":2.8, "revenueGrowth5Y":14.8,"dividendYield":1.4},
]

def _compute_score(roe, pe, de, margin):
    s = 0
    if roe >= 20: s += 30
    elif roe >= 15: s += 20
    elif roe >= 10: s += 10
    if pe and 0 < pe <= 20: s += 25
    elif pe and 0 < pe <= 35: s += 15
    if de < 0.5: s += 20
    elif de < 1: s += 10
    if margin >= 15: s += 25
    elif margin >= 8: s += 15
    return s

# Initialise cache from seed — screener works immediately with these
_SCREENER_CACHE: list = [{
    **s, "screenerSlug": s["symbol"], "patGrowth5Y": 0.0, "promoterHolding": 0.0,
    "score": _compute_score(s["roe"], s.get("pe") or 0, s["debtToEquity"], s["netMargin"])
} for s in _SEED]

_CACHE_LOCK    = _threading.Lock()
_REFRESH_LOCK  = _threading.Lock()
_LAST_REFRESH  = 0.0
_REFRESHING    = False
_QUERY_CACHE: dict = {}
_QUERY_TS: dict    = {}
_QUERY_TTL         = 1800  # 30 min


def _fetch_twelve_data():
    """
    Fetch fundamentals for all NSE stocks from Twelve Data API.
    Free tier: 800 credits/day → ~400 stocks (each /statistics call = 2 credits).
    Runs once daily in background. Falls back silently to seed data.
    """
    global _SCREENER_CACHE, _LAST_REFRESH, _REFRESHING
    if not TWELVE_DATA_KEY:
        print("[ROBU] screener: TWELVE_DATA_API_KEY not set — using seed data only")
        with _REFRESH_LOCK: _REFRESHING = False
        return

    try:
        print(f"[ROBU] screener: fetching from Twelve Data API...")
        base = "https://api.twelvedata.com"
        updated = {s["symbol"]: dict(s) for s in _SCREENER_CACHE}  # start from seed

        # Get full NSE stock list from Twelve Data
        resp = requests.get(f"{base}/stocks", params={"exchange":"NSE","country":"India","type":"Common Stock","outputsize":"5000"}, timeout=15)
        if not resp.ok:
            raise Exception(f"stocks list failed: {resp.status_code}")
        stocks_list = resp.json().get("data", [])
        print(f"[ROBU] Twelve Data: {len(stocks_list)} NSE stocks found")

        # Add any new symbols from TD to our STOCK_UNIVERSE
        for st in stocks_list:
            sym = st.get("symbol", "").replace(":NSE","").upper()
            if sym and sym not in updated:
                updated[sym] = {"symbol":sym, "name":st.get("name",sym), "screenerSlug":sym,
                    "sector":"NSE Listed", "price":0, "marketCap":0, "pe":None,
                    "roe":0, "roce":0, "netMargin":0, "debtToEquity":0,
                    "revenueGrowth5Y":0, "patGrowth5Y":0, "dividendYield":0,
                    "promoterHolding":0, "score":0}

        # Fetch fundamentals in batches — respect rate limits
        # /statistics endpoint: 2 credits per call
        # Free tier: 800 credits/day = 400 stocks max per day
        symbols_to_fetch = list(updated.keys())[:400]
        BATCH = 8  # 8 requests/minute rate limit on free tier
        for i in range(0, len(symbols_to_fetch), BATCH):
            batch = symbols_to_fetch[i:i+BATCH]
            for sym in batch:
                try:
                    td_sym = f"{sym}:NSE"
                    r = requests.get(f"{base}/statistics", params={"symbol":td_sym,"apikey":TWELVE_DATA_KEY}, timeout=10)
                    if not r.ok: continue
                    d = r.json()
                    if d.get("status") == "error": continue

                    # Correct Twelve Data /statistics response structure:
                    # d["statistics"]["valuations_metrics"], ["financials"], ["stock_statistics"]
                    # Values are plain strings/numbers — NOT {"value": ...} objects
                    stats = d.get("statistics", {})
                    vs = stats.get("valuations_metrics", {})
                    fs = stats.get("financials", {})
                    ss = stats.get("stock_statistics", {})

                    def _f(v): return float(v) if v not in (None, "", "N/A", "-") else 0.0

                    pe     = _f(vs.get("trailing_pe") or vs.get("forward_pe")) or None
                    mktcap = _f(ss.get("market_capitalization", 0)) / 1e7  # → ₹ Crore
                    roe    = _f(fs.get("return_on_equity_ttm")) * 100
                    margin = _f(fs.get("profit_margin")) * 100
                    de     = _f(fs.get("total_debt_to_equity_mrq"))
                    rev_g  = _f(fs.get("quarterly_revenue_growth_yoy")) * 100
                    div_y  = _f(ss.get("forward_annual_dividend_yield")) * 100

                    # Get current price from quote endpoint (1 credit)
                    qr = requests.get(f"{base}/price", params={"symbol":td_sym,"apikey":TWELVE_DATA_KEY}, timeout=8)
                    price = float(qr.json().get("price",0)) if qr.ok else updated.get(sym,{}).get("price",0)

                    if price > 0 or mktcap > 0:
                        updated[sym].update({"price":round(price,2),"marketCap":round(mktcap,1),
                            "pe":round(pe,1) if pe else None,"roe":round(roe,1),
                            # Twelve Data doesn't provide ROCE — don't fabricate it as
                            # roe*0.9 (ROCE and ROE diverge 2-3x in practice). Leave null.
                            "roce":None,"netMargin":round(margin,1),
                            "debtToEquity":round(de,2),"revenueGrowth5Y":round(rev_g,1),
                            "dividendYield":round(div_y,2),
                            "score":_compute_score(roe,pe or 0,de,margin)})
                except Exception: continue
            time.sleep(7)  # 8 requests/min → wait 7s between batches

            # Write partial updates progressively
            with _CACHE_LOCK:
                _SCREENER_CACHE = [v for v in updated.values() if v.get("price",0) > 0 or v["symbol"] in {s["symbol"] for s in _SEED}]

        with _CACHE_LOCK:
            _SCREENER_CACHE = [v for v in updated.values()]
            _LAST_REFRESH = time.time()
        print(f"[ROBU] Twelve Data refresh complete: {len(_SCREENER_CACHE)} stocks")

    except Exception as e:
        print(f"[ROBU] Twelve Data refresh error: {e} — seed data still serving")
    finally:
        with _REFRESH_LOCK: _REFRESHING = False


# Start background refresh if API key exists
if TWELVE_DATA_KEY:
    _REFRESHING = True
    _threading.Thread(target=_fetch_twelve_data, daemon=True).start()
else:
    print("[ROBU] screener: No TWELVE_DATA_API_KEY — serving 50 seed stocks. Add key to Railway env vars for live data.")


@app.get("/screener-v2")
def stock_screener_v2(
    min_roe:float=0, max_pe:float=9999, min_net_margin:float=-999,
    max_debt_equity:float=9999, min_market_cap:float=0, max_market_cap:float=9999999,
    min_rev_growth:float=-999, min_roce:float=0, sector:str="",
    sort_by:str="marketCap", order:str="desc", limit:int=60,
):
    global _REFRESHING
    # Trigger daily refresh
    if not _REFRESHING and TWELVE_DATA_KEY and time.time() - _LAST_REFRESH > 86400:
        with _REFRESH_LOCK:
            if not _REFRESHING:
                _REFRESHING = True
                _threading.Thread(target=_fetch_twelve_data, daemon=True).start()

    # Query cache
    qkey = _hashlib.md5(f"{min_roe}{max_pe}{min_net_margin}{max_debt_equity}{min_market_cap}{max_market_cap}{min_rev_growth}{min_roce}{sector}{sort_by}{order}{limit}".encode()).hexdigest()
    if qkey in _QUERY_CACHE and time.time() - _QUERY_TS.get(qkey,0) < _QUERY_TTL:
        return _QUERY_CACHE[qkey]

    with _CACHE_LOCK:
        data = list(_SCREENER_CACHE)

    filtered = []
    for s in data:
        if s.get("roe",0) < min_roe: continue
        if max_pe < 9999 and (not s.get("pe") or s["pe"] > max_pe): continue
        if s.get("netMargin",0) < min_net_margin: continue
        if max_debt_equity < 9999 and s.get("debtToEquity",0) > max_debt_equity: continue
        mc = s.get("marketCap",0)
        if mc > 0 and mc < min_market_cap: continue
        if mc > 0 and mc > max_market_cap: continue
        if min_rev_growth > -999 and s.get("revenueGrowth5Y",0) < min_rev_growth: continue
        if min_roce > 0 and s.get("roce",0) < min_roce: continue
        if sector and sector.lower() not in (s.get("sector","")).lower(): continue
        filtered.append(s)

    sk = {"Market Capitalization":"marketCap","Return on equity":"roe","Price to Earning":"pe",
          "Net profit margin":"netMargin","Sales growth 5Years":"revenueGrowth5Y",
          "Return on capital employed":"roce","marketCap":"marketCap","roe":"roe","score":"score"}.get(sort_by,"marketCap")
    filtered.sort(key=lambda x:(x.get(sk) or 0), reverse=(order=="desc"))
    result = filtered[:limit]

    _QUERY_CACHE[qkey] = result
    _QUERY_TS[qkey] = time.time()
    print(f"[ROBU] screener: {len(result)} results (cache: {len(data)} stocks, api_key: {'yes' if TWELVE_DATA_KEY else 'no'})")
    return result


@app.get("/screener-status")
def screener_status():
    with _CACHE_LOCK: count = len(_SCREENER_CACHE)
    return {"cached_stocks": count, "seed_stocks": len(_SEED),
            "api_key_set": bool(TWELVE_DATA_KEY), "refreshing": _REFRESHING,
            "last_refresh_hours": round((time.time()-_LAST_REFRESH)/3600,1) if _LAST_REFRESH else None}


# ---------------------------------------------------------------------------
# NSE Announcements — free public data from NSE India
# ---------------------------------------------------------------------------
@app.get("/announcements/{symbol}")
def get_announcements(symbol: str, limit: int = 10):
    """
    Fetch recent corporate announcements for a stock from NSE India.
    Uses NSE's public corporate filing endpoint — no auth needed.
    """
    symbol = symbol.upper().strip()
    cache_key = f"announcements:{symbol}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    headers = _make_browser_headers("https://www.nseindia.com/")
    headers["Referer"] = "https://www.nseindia.com/companies-listing/corporate-filings-announcements"

    try:
        # NSE public announcements API
        url = f"https://www.nseindia.com/api/corp-info?symbol={symbol}&subject=Corp%20Announcement"
        resp = requests.get(url, headers=headers, timeout=12)
        if not resp.ok:
            # Try alternate endpoint
            url2 = f"https://www.nseindia.com/api/corporates-announcements?index=equities&symbol={symbol}&issuer=&from_date=&to_date="
            resp = requests.get(url2, headers=headers, timeout=12)

        if resp.ok:
            data = resp.json()
            # NSE returns different shapes — normalize
            items = []
            raw = data if isinstance(data, list) else data.get("data", data.get("announcements", []))
            for item in raw[:limit]:
                items.append({
                    "date":     item.get("an_dt") or item.get("exchdisstime") or item.get("date", ""),
                    "subject":  item.get("subject") or item.get("desc") or item.get("subject", "Corporate Announcement"),
                    "category": item.get("sm_name") or item.get("attchmntText") or item.get("category", ""),
                    "url":      f"https://www.nseindia.com/companypage/{symbol}-announcements",
                })
            result = {"symbol": symbol, "announcements": items, "source": "nse"}
            _cache_set(cache_key, result)
            return result
    except Exception as e:
        print(f"[NSE announcements] {symbol}: {e}")

    # Fallback — return empty so frontend handles gracefully
    return {"symbol": symbol, "announcements": [], "source": "nse", "error": "unavailable"}



# ===========================================================================
# ROBU Discovery Engine — wiring (self-contained package in ./discovery)
# ---------------------------------------------------------------------------
# Heavy work runs as a nightly batch inside this package and is stored in
# SQLite; the /discovery endpoints just read it. See the discovery/ package
# and ROBU_Discovery_Engine_Architecture.docx for the full design.
# ===========================================================================
# Discovery is PARKED (removed from the live app 2026-06-14). It still spams
# Gemini HTTP 429 quota errors in the deploy logs via its boot/batch run, so the
# wiring is gated OFF by default. To revive: set env DISCOVERY_ENABLED=true.
if os.environ.get("DISCOVERY_ENABLED", "").lower() == "true":
    try:
        from discovery import init_discovery, DiscoveryDeps

        def _discovery_universe():
            with _CACHE_LOCK:
                return list(_SCREENER_CACHE)

        _discovery_deps = DiscoveryDeps(
            get_universe=_discovery_universe,
            fetch_company=company_v2,
            fetch_financials=financials,
            fetch_announcements=get_announcements,
            fetch_historical=historical_valuation,
            gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
            data_dir=os.path.dirname(os.path.abspath(__file__)),
            max_candidates=int(os.environ.get("DISCOVERY_MAX_CANDIDATES", "60")),
        )
        init_discovery(app, _discovery_deps)
        print("[ROBU] Discovery Engine mounted at /discovery")
    except Exception as _disc_err:
        print(f"[ROBU] Discovery Engine NOT mounted: {_disc_err}")
else:
    print("[ROBU] Discovery Engine disabled (set DISCOVERY_ENABLED=true to enable)")
