"""Scoring engine — blend the agent panel into a DiscoveryRecord.

Weights follow the architecture plan:
  Future Readiness   20%   (futureTrends)
  Industry Transform 18%   (industry)
  Financial quality  18%   (financial)
  Narrative Shift    16%   (news + market, avg)
  Smart-money/mispr. 14%   (market)
  Management quality 14%   (management)
  Skeptic penalty    -0..25 (subtracted after weighting)
"""

from __future__ import annotations
from typing import Dict, Any, List
from .models import DiscoveryRecord, grade_from_score, conviction_from

WEIGHTS = {
    "futureTrends": 0.20,
    "industry":     0.18,
    "financial":    0.18,
    "narrative":    0.16,   # avg(news, market)
    "market":       0.14,
    "management":   0.14,
}


def _i(d: Dict[str, Any], key: str, sub: str = "score", default: int = 50) -> int:
    try:
        v = d.get(key, {}).get(sub, default)
        return int(round(float(v)))
    except (TypeError, ValueError, AttributeError):
        return default


def _clean_list(v: Any, fallback: List[str]) -> List[str]:
    if isinstance(v, list):
        out = [str(x).strip() for x in v if str(x).strip()]
        if out:
            return out[:4]
    return fallback


def assign_category(stock: Dict[str, Any], signals: Dict[str, float],
                    panel: Dict[str, Any]) -> str:
    """Rules layer that maps sub-scores to one of the spec's categories."""
    q = signals
    fin = _i(panel, "financial")
    fut = _i(panel, "futureTrends")
    mkt = _i(panel, "market")
    growth = q["growth"]
    val = q["valuation"]
    under = q["underFollowed"]

    # Turnaround: weak-ish quality but improving / cheap
    if fin < 55 and (growth > 55 or val > 65):
        return "Turnaround"
    # Deep Value: cheap + decent quality, low growth
    if val >= 75 and growth < 45:
        return "Deep Value"
    # Future Multibagger: high future readiness + growth + smaller cap
    if fut >= 75 and growth >= 60 and under >= 50:
        return "Future Multibagger"
    # Emerging Leader: strong future + quality, mid cap
    if fut >= 65 and fin >= 60:
        return "Emerging Leader"
    # Capacity Expansion: strong growth signal
    if growth >= 70:
        return "Capacity Expansion"
    # Smart Money: market/flows favourable + mispricing
    if mkt >= 70:
        return "Smart Money"
    # Default: quietly compounding quality the crowd hasn't noticed
    return "Hidden Compounder"


def build_record(stock: Dict[str, Any], signals: Dict[str, float],
                 panel: Dict[str, Any], known_symbols: set) -> DiscoveryRecord:
    fin  = _i(panel, "financial")
    mgmt = _i(panel, "management")
    ind  = _i(panel, "industry")
    mkt  = _i(panel, "market")
    news = _i(panel, "news")
    fut  = _i(panel, "futureTrends")

    industry_transform = _i(panel, "industry", "transformationScore", ind)
    future_readiness   = _i(panel, "futureTrends", "readinessScore", fut)
    narrative = round((news + mkt) / 2)

    base = (
        WEIGHTS["futureTrends"] * fut +
        WEIGHTS["industry"]     * ind +
        WEIGHTS["financial"]    * fin +
        WEIGHTS["narrative"]    * narrative +
        WEIGHTS["market"]       * mkt +
        WEIGHTS["management"]   * mgmt
    )

    skeptic = panel.get("skeptic", {}) or {}
    try:
        penalty = max(0, min(25, int(round(float(skeptic.get("penalty", 8))))))
    except (TypeError, ValueError):
        penalty = 8

    score = int(round(max(0, min(100, base - penalty))))

    ft = panel.get("futureTrends", {}) or {}
    mk = panel.get("market", {}) or {}
    nw = panel.get("news", {}) or {}

    data_completeness = 0.9 if panel.get("_source") == "ai" else 0.45
    conviction = conviction_from(score, penalty, data_completeness)

    return DiscoveryRecord(
        symbol=stock.get("symbol", "?"),
        name=stock.get("name", stock.get("symbol", "?")),
        sector=stock.get("sector", "Unknown"),
        discoveryScore=score,
        grade=grade_from_score(score),
        category=assign_category(stock, signals, panel),
        aiConviction=conviction,
        isNew=stock.get("symbol") not in known_symbols,
        whyFound=str(panel.get("whyFound", "Surfaced on quality + growth screen."))[:400],
        whyNow=str(panel.get("whyNow", "Fundamentals and momentum align."))[:400],
        futureTailwinds=_clean_list(ft.get("tailwinds"), ["Sector demand growth"]),
        futureThreats=_clean_list(ft.get("threats"), ["Competitive intensity"]),
        hiddenOptionality=str(panel.get("hiddenOptionality", "Optionality on faster adoption."))[:300],
        narrativeShift=str(nw.get("narrativeShift", "Narrative re-rating potential."))[:300],
        whyMarketMayBeWrong=str(mk.get("whyMarketMayBeWrong", "Crowd may under-rate the setup."))[:400],
        keyRisks=_clean_list(skeptic.get("keyRisks"), ["Execution risk", "Valuation risk"]),
        industryTransformationScore=industry_transform,
        futureReadinessScore=future_readiness,
    )
