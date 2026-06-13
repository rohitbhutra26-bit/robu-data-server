"""Phase 3a — Real data feeds for the agents.

Two feeds:
  * News      — reused from main.py's /announcements (NSE corporate filings),
                passed in via deps.fetch_announcements.
  * FII/DII   — market-wide foreign & domestic institutional flows from NSE.
                Net-new here. Cached once per run.

All feeds fail soft: if a source is down the agent simply gets less context,
never an exception that breaks the nightly run.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Callable
import time

try:
    from curl_cffi import requests as _cffi
    _HAS_CFFI = True
except Exception:
    _HAS_CFFI = False
import requests as _rq


_BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

_FII_DII_CACHE: Dict[str, Any] = {}
_FII_DII_TS: float = 0.0
_FII_TTL = 6 * 3600  # 6 hours


def _get_json(url: str, referer: str, timeout: int = 12) -> Optional[Any]:
    headers = dict(_BROWSER_HEADERS)
    headers["Referer"] = referer
    try:
        if _HAS_CFFI:
            # NSE needs a real cookie jar; hit the home page first to get cookies.
            sess = _cffi.Session(impersonate="chrome124")
            sess.get("https://www.nseindia.com/", headers=headers, timeout=timeout)
            r = sess.get(url, headers=headers, timeout=timeout)
        else:
            sess = _rq.Session()
            sess.get("https://www.nseindia.com/", headers=headers, timeout=timeout)
            r = sess.get(url, headers=headers, timeout=timeout)
        if r.ok:
            return r.json()
    except Exception as e:
        print(f"[discovery.feeds] {url}: {e}")
    return None


def get_fii_dii() -> Dict[str, Any]:
    """Market-wide FII/DII net flows (₹ Cr) for the latest session.

    Returns a small context dict the Market agent can reason over:
        {"fiiNet": -1234.5, "diiNet": 2100.0, "tone": "DII absorbing FII selling"}
    """
    global _FII_DII_TS
    now = time.time()
    if _FII_DII_CACHE and now - _FII_DII_TS < _FII_TTL:
        return _FII_DII_CACHE

    out: Dict[str, Any] = {"fiiNet": None, "diiNet": None, "tone": "unknown", "source": "nse"}
    data = _get_json(
        "https://www.nseindia.com/api/fiidiiTradeReact",
        "https://www.nseindia.com/reports/fii-dii",
    )
    try:
        if isinstance(data, list):
            for row in data:
                cat = (row.get("category") or "").upper()
                net = row.get("netValue") or row.get("net")
                net = float(str(net).replace(",", "")) if net not in (None, "") else None
                if "FII" in cat or "FPI" in cat:
                    out["fiiNet"] = net
                elif "DII" in cat:
                    out["diiNet"] = net
            fii, dii = out["fiiNet"], out["diiNet"]
            if fii is not None and dii is not None:
                if fii < 0 and dii > 0:
                    out["tone"] = "DII buying, FII selling"
                elif fii > 0 and dii > 0:
                    out["tone"] = "both buying (risk-on)"
                elif fii < 0 and dii < 0:
                    out["tone"] = "both selling (risk-off)"
                else:
                    out["tone"] = "FII buying, DII selling"
    except Exception as e:
        print(f"[discovery.feeds] fii/dii parse: {e}")

    _FII_DII_CACHE.update(out)
    _FII_DII_TS = now
    return out


def get_news(symbol: str, fetch_announcements: Optional[Callable], limit: int = 6) -> List[Dict[str, str]]:
    """Recent corporate announcements for one stock (reuses main.py feed)."""
    if not fetch_announcements:
        return []
    try:
        res = fetch_announcements(symbol, limit) if _accepts_two(fetch_announcements) else fetch_announcements(symbol)
        items = (res or {}).get("announcements", []) if isinstance(res, dict) else []
        return items[:limit]
    except Exception as e:
        print(f"[discovery.feeds] news {symbol}: {e}")
        return []


def _accepts_two(fn: Callable) -> bool:
    try:
        import inspect
        return len(inspect.signature(fn).parameters) >= 2
    except Exception:
        return False
