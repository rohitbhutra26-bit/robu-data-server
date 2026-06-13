"""Phase 4 — Orchestrator. Ties the pipeline together for one nightly run.

    universe ─▶ select_candidates ─▶ for each: news + panel + score ─▶ store

Cost guard: only deps.max_candidates stocks reach the AI panel. FII/DII is
fetched once per run, not per stock. Everything fails soft.
"""

from __future__ import annotations
from typing import Dict, Any, List
import threading
import traceback

from .deps import DiscoveryDeps
from .quant import select_candidates, quant_signals
from .feeds import get_fii_dii, get_news
from .agents import run_panel
from .scoring import build_record
from .store import DiscoveryStore

_RUN_LOCK = threading.Lock()
_RUNNING = False


def is_running() -> bool:
    return _RUNNING


def run_once(deps: DiscoveryDeps, store: DiscoveryStore) -> Dict[str, Any]:
    """Execute one full discovery run (blocking). Returns a summary."""
    global _RUNNING
    if not _RUN_LOCK.acquire(blocking=False):
        return {"ok": False, "reason": "a run is already in progress"}
    _RUNNING = True
    run_id = store.start_run()
    sources = set()
    try:
        universe = deps.get_universe() or []
        candidates = select_candidates(
            universe, deps.max_candidates, deps.min_market_cap_cr, deps.max_market_cap_cr
        )
        if not candidates:
            store.fail_run(run_id, "empty universe")
            return {"ok": False, "reason": "empty universe"}

        fiidii = get_fii_dii()
        known = store.previous_symbols()

        records: List[Dict[str, Any]] = []
        prices: Dict[str, float] = {}
        for stock in candidates:
            try:
                sym = stock.get("symbol")
                signals = quant_signals(stock)
                news = get_news(sym, deps.fetch_announcements)
                panel = run_panel(stock, signals, news, fiidii, deps.gemini_api_key)
                sources.add(panel.get("_source", "?"))
                rec = build_record(stock, signals, panel, known)
                records.append(rec.to_dict())
                try:
                    prices[sym] = float(stock.get("price") or 0) or None
                except (TypeError, ValueError):
                    prices[sym] = None
            except Exception as e:
                print(f"[discovery.orchestrator] {stock.get('symbol')}: {e}")
                continue

        records.sort(key=lambda r: r["discoveryScore"], reverse=True)
        source = "ai" if "ai" in sources else "fallback"
        store.finish_run(run_id, records, prices, source)
        print(f"[discovery] run {run_id} done: {len(records)} ideas (source={source})")
        return {"ok": True, "runId": run_id, "count": len(records), "source": source}

    except Exception as e:
        traceback.print_exc()
        store.fail_run(run_id, str(e))
        return {"ok": False, "reason": str(e)}
    finally:
        _RUNNING = False
        _RUN_LOCK.release()


def run_in_background(deps: DiscoveryDeps, store: DiscoveryStore) -> None:
    t = threading.Thread(target=run_once, args=(deps, store), daemon=True)
    t.start()
