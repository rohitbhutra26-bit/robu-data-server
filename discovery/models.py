"""Data models for the Discovery Engine.

DiscoveryRecord field names match the TypeScript interface in
robu-valuation-next/src/app/api/discovery/route.ts EXACTLY (camelCase),
so the JSON the API serves drops straight into the UI with no mapping.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List
from datetime import datetime, timezone

# The seven discovery categories (mirrors the spec).
CATEGORIES = [
    "Hidden Compounder",
    "Turnaround",
    "Emerging Leader",
    "Capacity Expansion",
    "Deep Value",
    "Smart Money",
    "Future Multibagger",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class DiscoveryRecord:
    symbol: str
    name: str
    sector: str
    discoveryScore: int            # 0-100
    grade: str                     # A+ ... F
    category: str
    aiConviction: str              # High | Medium | Low
    isNew: bool
    whyFound: str
    whyNow: str
    futureTailwinds: List[str]
    futureThreats: List[str]
    hiddenOptionality: str
    narrativeShift: str
    whyMarketMayBeWrong: str
    keyRisks: List[str]
    industryTransformationScore: int
    futureReadinessScore: int
    generatedAt: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgentOutput:
    """Standard envelope every agent returns."""
    score: int                     # 0-100 for this dimension
    reason: str                    # one-line justification
    extra: dict = field(default_factory=dict)  # agent-specific narrative fields


def grade_from_score(score: int) -> str:
    if score >= 90: return "A+"
    if score >= 80: return "A"
    if score >= 72: return "A-"
    if score >= 65: return "B+"
    if score >= 55: return "B"
    if score >= 45: return "C"
    if score >= 35: return "D"
    return "F"


def conviction_from(score: int, skeptic_penalty: int, data_completeness: float) -> str:
    """High conviction needs a high score, a low skeptic penalty and good data."""
    if score >= 78 and skeptic_penalty <= 10 and data_completeness >= 0.7:
        return "High"
    if score >= 60 and data_completeness >= 0.5:
        return "Medium"
    return "Low"
