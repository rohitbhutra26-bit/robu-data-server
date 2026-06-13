"""FastAPI router for the Discovery Engine.

Endpoints (all read from the store except /run which triggers a batch):
  GET  /discovery            -> latest ranked feed (optional ?category=)
  GET  /discovery/status     -> last run info + scheduler state
  GET  /discovery/backtest   -> Phase 5 score-vs-return check
  POST /discovery/run        -> trigger a batch run in the background (guarded)
"""

from __future__ import annotations
from fastapi import APIRouter, Header, HTTPException
from typing import Optional
import os

from .deps import DiscoveryDeps
from .store import DiscoveryStore
from .orchestrator import run_in_background, is_running
from .backtest import run_backtest


def make_router(deps: DiscoveryDeps, store: DiscoveryStore) -> APIRouter:
    router = APIRouter(prefix="/discovery", tags=["discovery"])
    run_token = os.environ.get("DISCOVERY_RUN_TOKEN", "")

    @router.get("")
    def get_discovery(category: Optional[str] = None):
        records = store.latest_records(category)
        run = store.latest_run()
        new_count = sum(1 for r in records if r.get("isNew"))
        return {
            "generatedAt": records[0]["generatedAt"] if records else (run["run_ts"] if run else None),
            "newCount": new_count,
            "total": len(records),
            "records": records,
            "running": is_running(),
        }

    @router.get("/status")
    def get_status():
        st = store.status()
        st["running"] = is_running()
        st["maxCandidates"] = deps.max_candidates
        st["aiEnabled"] = bool(deps.gemini_api_key)
        return st

    @router.get("/backtest")
    def get_backtest():
        return run_backtest(store)

    @router.post("/run")
    def trigger_run(x_run_token: Optional[str] = Header(default=None)):
        # If a token is configured, require it (protects the paid AI batch).
        if run_token and x_run_token != run_token:
            raise HTTPException(403, "invalid run token")
        if is_running():
            return {"ok": False, "reason": "already running"}
        run_in_background(deps, store)
        return {"ok": True, "started": True}

    return router
