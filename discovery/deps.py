"""Dependency container.

The Discovery package never imports from main.py (avoids circular imports).
Instead main.py passes in the data accessors it already owns via this struct.
Every callable is optional/defensive so the package still runs if one source
is down.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Dict, Any, Optional


@dataclass
class DiscoveryDeps:
    # Returns a snapshot list of universe stocks (the screener cache).
    get_universe: Callable[[], List[Dict[str, Any]]]

    # Per-symbol fetchers (reuse existing endpoint functions). May raise.
    fetch_company: Optional[Callable[[str], Dict[str, Any]]] = None
    fetch_financials: Optional[Callable[[str], Any]] = None
    fetch_announcements: Optional[Callable[..., Dict[str, Any]]] = None
    fetch_historical: Optional[Callable[[str], Dict[str, Any]]] = None

    # Config
    gemini_api_key: str = ""
    data_dir: str = "."
    max_candidates: int = 60          # cost guard: stocks that reach the agents
    min_market_cap_cr: float = 1000   # ignore micro/illiquid names
    max_market_cap_cr: float = 400000 # discovery favours the under-followed
