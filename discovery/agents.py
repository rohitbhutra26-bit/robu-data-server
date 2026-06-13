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


def _fallback(stock: Dict[str, Any], signals: Dict[str, float],
              news: List[Dict[str, str]], fiidii: Dict[str, Any]) -> Dict[str, Any]:
    """No-AI synthesis from quant signals so the engine still works offline."""
    q = signals
    fin = int(round(0.6 * q["quality"] + 0.4 * q["growth"]))
    ind = int(round(0.5 * q["growth"] + 0.5 * q["quality"]))
    mkt = int(round(0.6 * q["valuation"] + 0.4 * q["underFollowed"]))
    fut = int(round(0.5 * q["growth"] + 0.5 * q["underFollowed"]))
    sector = (stock.get("sector") or "").lower()
    theme = next((t for t in TREND_THEMES if t.split()[0].lower() in sector), "Specialty Manufacturing")
    recent = news[0]["subject"][:120] if news else "steady operations, no major surprises"
    return {
        "financial":   {"score": fin, "reason": f"ROE {stock.get('roe','?')}%, 5Y PAT growth {stock.get('patGrowth5Y','?')}%."},
        "management":  {"score": int(round(0.5 * (fin + q['quality']))), "reason": "Inferred from return ratios and consistency."},
        "industry":    {"score": ind, "reason": f"{stock.get('sector','Sector')} momentum from growth profile.", "transformationScore": ind},
        "market":      {"score": mkt, "reason": f"P/E {stock.get('pe','?')}x vs growth; flows {fiidii.get('tone')}.",
                         "whyMarketMayBeWrong": "Crowd may under-rate the growth-to-valuation gap."},
        "news":        {"score": min(70, mkt), "reason": recent, "narrativeShift": "Awaiting a clear catalyst in filings."},
        "futureTrends":{"score": fut, "reason": f"Exposure to {theme}.", "readinessScore": fut,
                         "tailwinds": [f"{theme} demand", "Operating leverage on growth"], "threats": ["Competitive intensity"], "theme": theme},
        "skeptic":     {"penalty": 8 if q["valuation"] < 40 else 4,
                         "keyRisks": ["Valuation leaves little margin for error" if q["valuation"] < 40 else "Execution risk on growth",
                                      "Cyclical or demand sensitivity"]},
        "whyFound": f"{stock.get('name')} screens well on quality and growth while still {('a smaller, under-followed name' if q['underFollowed']>50 else 'reasonably valued')}.",
        "whyNow": "Fundamentals and momentum line up before broad recognition.",
        "hiddenOptionality": f"Upside if {theme.lower()} adoption accelerates faster than priced in.",
    }
