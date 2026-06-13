"""Phase 2 — Quant funnel + base scoring.

Two jobs:
  1. select_candidates() — cheaply rank the WHOLE universe and keep the top N.
     This is the cost guard: only these stocks ever reach the (paid) AI agents.
  2. quant_signals() — turn one stock's fundamentals into 0-100 sub-signals
     that seed the Financial / Industry dimensions and the category rules.

Everything here uses fields already present in the screener cache, so it is
free and fast (no network calls).
"""

from __future__ import annotations
from typing import Dict, Any, List


def _f(stock: Dict[str, Any], key: str, default: float = 0.0) -> float:
    v = stock.get(key, default)
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def quality_score(stock: Dict[str, Any]) -> float:
    """How good is the business? ROE/ROCE/margin/leverage."""
    roe   = _f(stock, "roe")
    roce  = _f(stock, "roce")
    de    = _f(stock, "debtToEquity")
    marg  = _f(stock, "netMargin")
    s = 0.0
    s += _clamp(roe * 1.6, 0, 35)          # 22%+ ROE ~ full marks
    s += _clamp(roce * 1.2, 0, 30)
    s += _clamp(marg * 1.2, 0, 20)
    s += 15 if de < 0.5 else (8 if de < 1.0 else 0)
    return _clamp(s)


def growth_score(stock: Dict[str, Any]) -> float:
    """Top- and bottom-line momentum."""
    rev = _f(stock, "revenueGrowth5Y")
    pat = _f(stock, "patGrowth5Y")
    # PAT growth weighted higher; cap each contribution
    s = _clamp(rev * 1.8, 0, 45) + _clamp(pat * 1.6, 0, 55)
    return _clamp(s)


def valuation_attractiveness(stock: Dict[str, Any]) -> float:
    """Cheap-ish relative to quality? High score = more attractively priced."""
    pe = _f(stock, "pe")
    if pe <= 0:
        return 40.0  # loss-making / NA — neutral-ish, let agents judge
    if pe <= 15: return 100.0
    if pe <= 25: return 80.0
    if pe <= 35: return 60.0
    if pe <= 50: return 40.0
    if pe <= 70: return 25.0
    return 12.0


def under_followed_bonus(stock: Dict[str, Any]) -> float:
    """Discovery favours names the crowd hasn't piled into yet.
    Smaller (but not micro) caps get a bonus; mega caps get little."""
    mcap = _f(stock, "marketCap")
    if mcap <= 0:
        return 0.0
    if mcap < 20000:   return 100.0
    if mcap < 50000:   return 75.0
    if mcap < 100000:  return 50.0
    if mcap < 200000:  return 25.0
    return 8.0


def discovery_prelim(stock: Dict[str, Any]) -> float:
    """Composite used ONLY to rank/shortlist the universe before AI runs.
    Blends quality, growth, valuation and 'under-followed' optionality."""
    q = quality_score(stock)
    g = growth_score(stock)
    v = valuation_attractiveness(stock)
    u = under_followed_bonus(stock)
    return round(0.34 * q + 0.30 * g + 0.18 * v + 0.18 * u, 1)


def select_candidates(
    universe: List[Dict[str, Any]],
    top_n: int,
    min_mcap: float,
    max_mcap: float,
) -> List[Dict[str, Any]]:
    """Filter by liquidity band, then keep the top_n by prelim score.
    Sector-diversified: cap any single sector at ~35% of the shortlist so the
    feed isn't all IT or all banks."""
    pool = []
    for s in universe:
        mcap = _f(s, "marketCap")
        if mcap < min_mcap or mcap > max_mcap:
            continue
        s = dict(s)
        s["_prelim"] = discovery_prelim(s)
        pool.append(s)

    pool.sort(key=lambda x: x["_prelim"], reverse=True)

    sector_cap = max(3, int(top_n * 0.35))
    sector_count: Dict[str, int] = {}
    picked: List[Dict[str, Any]] = []
    for s in pool:
        sec = (s.get("sector") or "Unknown")
        if sector_count.get(sec, 0) >= sector_cap:
            continue
        picked.append(s)
        sector_count[sec] = sector_count.get(sec, 0) + 1
        if len(picked) >= top_n:
            break
    return picked


def quant_signals(stock: Dict[str, Any]) -> Dict[str, float]:
    """Sub-signals consumed by the scoring engine and agents."""
    return {
        "quality": round(quality_score(stock), 1),
        "growth": round(growth_score(stock), 1),
        "valuation": round(valuation_attractiveness(stock), 1),
        "underFollowed": round(under_followed_bonus(stock), 1),
        "prelim": round(discovery_prelim(stock), 1),
    }
