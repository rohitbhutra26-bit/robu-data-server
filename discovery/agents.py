"""Phase 3b — The 7 agents.

Conceptually seven specialists, each answering one question and returning a
0-100 sub-score + narrative. For cost/latency we execute them as ONE structured
Gemini call per stock (the prompt assigns all seven roles and demands a single
JSON object keyed by agent). A deterministic fallback runs when no API key is
set or the call fails, so the pipeline always produces records.
"""

from __future__ import annotations
from typing import Dict, Any, List
from .gemini import gemini_json

# Spec's future-trend themes for the Future Trends agent.
TREND_THEMES = [
    "Artificial Intelligence", "Automation", "Robotics", "Data Centers",
    "Semiconductors", "EV Ecosystem", "Energy Transition",
    "Defense Modernization", "Digital Infrastructure", "Specialty Manufacturing",
]

AGENT_KEYS = ["financial", "management", "industry", "market", "news", "futureTrends", "skeptic"]


def _fmt_news(news: List[Dict[str, str]]) -> str:
    if not news:
        return "No recent filings retrieved."
    lines = []
    for n in news[:5]:
        subj = (n.get("subject") or "").strip()[:140]
        date = (n.get("date") or "").strip()[:20]
        if subj:
            lines.append(f"- {date}: {subj}")
    return "\n".join(lines) or "No recent filings retrieved."


def build_panel_prompt(stock: Dict[str, Any], signals: Dict[str, float],
                       news: List[Dict[str, str]], fiidii: Dict[str, Any]) -> str:
    name = stock.get("name", stock.get("symbol", "?"))
    return f"""You are ROBU, a panel of seven senior equity-research specialists at an Indian institutional fund. You are screening for stocks the broader market has NOT yet fully recognised. Be specific and cite numbers. Avoid generic phrases like "strong fundamentals" or "well-positioned".

COMPANY: {name} ({stock.get('symbol')})
SECTOR: {stock.get('sector','Unknown')}
PRICE: Rs {stock.get('price','?')} | MARKET CAP: Rs {stock.get('marketCap','?')} Cr
RATIOS: P/E {stock.get('pe','?')}x | ROE {stock.get('roe','?')}% | ROCE {stock.get('roce','?')}% | Net Margin {stock.get('netMargin','?')}% | D/E {stock.get('debtToEquity','?')} | Div Yield {stock.get('dividendYield','?')}%
GROWTH (5Y): Revenue {stock.get('revenueGrowth5Y','?')}% | PAT {stock.get('patGrowth5Y','?')}%
QUANT SIGNALS (0-100): quality {signals.get('quality')} | growth {signals.get('growth')} | valuation {signals.get('valuation')} | under-followed {signals.get('underFollowed')}

RECENT FILINGS / NEWS:
{_fmt_news(news)}

MARKET FLOWS: FII net Rs {fiidii.get('fiiNet')} Cr, DII net Rs {fiidii.get('diiNet')} Cr ({fiidii.get('tone')}).

FUTURE-TREND THEMES to check the company against: {", ".join(TREND_THEMES)}.

Each specialist returns a score 0-100 (higher = more attractive for discovery) and one specific reason:
1. financial   — Is the money engine healthy AND improving? Use ROE/ROCE/margin/growth.
2. management  — Do they execute and invest for the future? Infer from filings/capex signals.
3. industry    — Is the whole industry inflecting up? Give an industry transformation read.
4. market      — Is it mispriced and is smart money positioning? Use valuation + flows.
5. news        — What recently changed that matters? Cite a filing if any.
6. futureTrends— Which of the listed themes does it ride, and how ready is it?
7. skeptic     — Red-team the above. 'penalty' 0-25 = how much to dock for real risks.

Return ONLY this JSON (no markdown):
{{
  "financial":   {{"score": 0, "reason": ""}},
  "management":  {{"score": 0, "reason": ""}},
  "industry":    {{"score": 0, "reason": "", "transformationScore": 0}},
  "market":      {{"score": 0, "reason": "", "whyMarketMayBeWrong": ""}},
  "news":        {{"score": 0, "reason": "", "narrativeShift": ""}},
  "futureTrends":{{"score": 0, "reason": "", "readinessScore": 0, "tailwinds": ["",""], "threats": [""], "theme": ""}},
  "skeptic":     {{"penalty": 0, "keyRisks": ["",""]}},
  "whyFound": "1-2 sentences: why ROBU surfaced this now.",
  "whyNow": "1-2 sentences: why the timing is interesting.",
  "hiddenOptionality": "one line: an upside the market ignores."
}}"""


def run_panel(stock: Dict[str, Any], signals: Dict[str, float],
              news: List[Dict[str, str]], fiidii: Dict[str, Any],
              api_key: str) -> Dict[str, Any]:
    """Run the agent panel; fall back to a deterministic synthesis on failure."""
    prompt = build_panel_prompt(stock, signals, news, fiidii)
    out = gemini_json(prompt, api_key)
    if out and _looks_valid(out):
        out["_source"] = "ai"
        return out
    fb = _fallback(stock, signals, news, fiidii)
    fb["_source"] = "fallback"
    return fb


def _looks_valid(out: Dict[str, Any]) -> bool:
    return all(k in out for k in ("financial", "industry", "market", "futureTrends", "skeptic"))


# Map real sectors → a relevant tailwind theme + a sector-specific risk, so the
# no-AI fallback is differentiated per stock instead of one generic template.
_SECTOR_PROFILE = {
    "technology":        ("Artificial Intelligence adoption", "AI could compress services pricing"),
    "software":          ("Enterprise AI & cloud spend", "AI disruption to legacy revenue"),
    "financial":         ("Credit growth & formalisation", "Asset-quality / credit-cycle risk"),
    "bank":              ("Credit growth & deposit franchise", "NPA / credit-cycle risk"),
    "healthcare":        ("Specialty & chronic-care demand", "USFDA / pricing regulation risk"),
    "pharma":            ("Complex generics & CDMO demand", "USFDA / price-erosion risk"),
    "automobile":        ("EV transition & premiumisation", "Demand cyclicality & input costs"),
    "auto":              ("EV ecosystem build-out", "Commodity & demand cyclicality"),
    "metals":            ("Infra & capex up-cycle", "Commodity-price cyclicality"),
    "mining":            ("Energy & base-load demand", "Energy-transition headwind"),
    "energy":            ("Energy transition & power demand", "Regulated pricing / policy risk"),
    "power":             ("Rising power demand & grid capex", "Regulated returns / fuel risk"),
    "oil":               ("Refining margins & gas demand", "Crude-price volatility"),
    "consumer":          ("Premiumisation & rural recovery", "Input-cost inflation, slow volumes"),
    "fmcg":              ("Distribution & premiumisation", "Volume growth & margin pressure"),
    "capital goods":     ("Capex & infrastructure up-cycle", "Order-execution & working-capital risk"),
    "defence":           ("Defence indigenisation orders", "Budget dependence & order lumpiness"),
    "chemical":          ("Specialty chemicals import-substitution", "China dumping / cyclicality"),
    "realty":            ("Housing up-cycle & consolidation", "Interest-rate & demand cyclicality"),
    "infrastructure":    ("Government infra capex", "Execution & leverage risk"),
    "services":          ("Formalisation & digital adoption", "Competitive intensity"),
    "communication":     ("Data consumption & tariff hikes", "Capex intensity & competition"),
}


def _sector_profile(sector: str) -> tuple:
    s = (sector or "").lower()
    for key, prof in _SECTOR_PROFILE.items():
        if key in s:
            return prof
    return ("Sector demand & operating leverage", "Competitive intensity")


def _fallback(stock: Dict[str, Any], signals: Dict[str, float],
              news: List[Dict[str, str]], fiidii: Dict[str, Any]) -> Dict[str, Any]:
    """No-AI synthesis from quant signals — differentiated per stock by its own
    numbers and sector, so cards never read identically."""
    q = signals
    fin = int(round(0.6 * q["quality"] + 0.4 * q["growth"]))
    ind = int(round(0.5 * q["growth"] + 0.5 * q["quality"]))
    mkt = int(round(0.6 * q["valuation"] + 0.4 * q["underFollowed"]))
    fut = int(round(0.5 * q["growth"] + 0.5 * q["underFollowed"]))

    name = stock.get("name", stock.get("symbol", "This company"))
    roe = stock.get("roe", "?"); roce = stock.get("roce", "?")
    pe = stock.get("pe", "?"); pat_g = stock.get("patGrowth5Y", "?")
    rev_g = stock.get("revenueGrowth5Y", "?"); de = stock.get("debtToEquity", 0)
    tailwind, sector_risk = _sector_profile(stock.get("sector", ""))

    # whyNow varies by the stock's actual profile.
    try:
        cheap = float(pe) > 0 and float(pe) <= 22
    except (TypeError, ValueError):
        cheap = False
    if q["underFollowed"] >= 60 and q["growth"] >= 55:
        whynow = f"{name} grows PAT ~{pat_g}% on {roe}% ROE yet is still a smaller, under-owned name — the kind the crowd finds late."
    elif cheap:
        whynow = f"At ~{pe}x earnings with {roce}% ROCE, {name} looks under-priced versus its {rev_g}% revenue growth."
    elif q["quality"] >= 65:
        whynow = f"{name} earns a high {roe}% ROE with {('low' if (de or 0) < 0.5 else 'manageable')} debt — durable quality the market under-rates."
    else:
        whynow = f"{name} shows improving momentum ({pat_g}% PAT growth) that hasn't fully shown up in the price yet."

    recent = news[0]["subject"][:130] if news else f"No major filings; thesis rests on {rev_g}% growth and {roe}% returns."
    leverage_risk = "High debt adds refinancing risk" if (de or 0) >= 1.0 else None
    val_risk = "Valuation leaves little margin for error" if q["valuation"] < 40 else None
    risks = [r for r in (val_risk, sector_risk, leverage_risk) if r][:3] or ["Execution risk on growth"]

    return {
        "financial":   {"score": fin, "reason": f"ROE {roe}%, ROCE {roce}%, 5Y PAT growth {pat_g}%."},
        "management":  {"score": int(round(0.5 * (fin + q['quality']))), "reason": f"Consistent {roe}% returns imply disciplined capital allocation."},
        "industry":    {"score": ind, "reason": f"{stock.get('sector','Sector')} riding {tailwind.lower()}.", "transformationScore": ind},
        "market":      {"score": mkt, "reason": f"P/E {pe}x vs {rev_g}% growth; market flows {fiidii.get('tone')}.",
                         "whyMarketMayBeWrong": f"Crowd anchors on the past and under-rates {name}'s growth-to-valuation gap."},
        "news":        {"score": min(70, mkt), "reason": recent, "narrativeShift": f"Re-rating likely as {tailwind.lower()} plays out."},
        "futureTrends":{"score": fut, "reason": f"Exposure to {tailwind}.", "readinessScore": fut,
                         "tailwinds": [tailwind, f"Operating leverage on {rev_g}% growth"], "threats": [sector_risk], "theme": tailwind},
        "skeptic":     {"penalty": 8 if q["valuation"] < 40 else 4, "keyRisks": risks},
        "whyFound": f"{name} scores well on quality (ROE {roe}%) and growth (PAT {pat_g}%) while still {('under-followed' if q['underFollowed']>50 else 'reasonably valued')}.",
        "whyNow": whynow,
        "hiddenOptionality": f"Upside if {tailwind.lower()} accelerates faster than the market prices in.",
    }
