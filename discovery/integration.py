"""One-call wiring for main.py.

    from discovery import init_discovery, DiscoveryDeps
    init_discovery(app, DiscoveryDeps(get_universe=lambda: list(_SCREENER_CACHE), ...))

Does three things:
  1. creates the SQLite store,
  2. mounts the /discovery router,
  3. starts a daily scheduler (and a one-off run on boot if the store is empty).

The scheduler is a plain daemon thread — no extra dependency. It wakes hourly
and runs the batch if it's past the configured hour and hasn't run today.
"""

from __future__ import annotations
from typing import Optional
import os
import time
import threading
from datetime import datetime

from .deps import DiscoveryDeps
from .store import DiscoveryStore
from .router import make_router
from .orchestrator import run_once, is_running


def _scheduler_loop(deps: DiscoveryDeps, store: DiscoveryStore, hour_utc: int) -> None:
    last_run_date: Optional[str] = None
    # Run on boot if there's no feed yet, OR if the last feed was the non-AI
    # fallback while an AI key is now available (so a deploy that fixes the
    # model automatically refreshes the feed with real AI narratives).
    last = store.latest_run()
    needs_boot = (last is None) or (
        deps.gemini_api_key and (last["source"] if last else "") != "ai"
    )
    if needs_boot:
        reason = "no prior feed" if last is None else "upgrading fallback feed to AI"
        print(f"[discovery] boot run — {reason}")
        try:
            run_once(deps, store)
            last_run_date = datetime.utcnow().strftime("%Y-%m-%d")
        except Exception as e:
            print(f"[discovery] boot run failed: {e}")

    while True:
        try:
            now = datetime.utcnow()
            today = now.strftime("%Y-%m-%d")
            if now.hour >= hour_utc and last_run_date != today and not is_running():
                print(f"[discovery] nightly trigger at {now.isoformat()}Z")
                run_once(deps, store)
                last_run_date = today
        except Exception as e:
            print(f"[discovery] scheduler: {e}")
        time.sleep(1800)  # check every 30 min


def init_discovery(app, deps: DiscoveryDeps, *, schedule_hour_utc: Optional[int] = None) -> DiscoveryStore:
    store = DiscoveryStore(os.path.join(deps.data_dir, "discovery_data"))
    app.include_router(make_router(deps, store))

    # Default nightly hour: 01:30 IST ≈ 20:00 UTC (after market close + data refresh).
    hour = schedule_hour_utc
    if hour is None:
        hour = int(os.environ.get("DISCOVERY_SCHEDULE_HOUR_UTC", "20"))

    enabled = os.environ.get("DISCOVERY_SCHEDULER", "on").lower() not in ("off", "false", "0")
    if enabled:
        t = threading.Thread(target=_scheduler_loop, args=(deps, store, hour), daemon=True)
        t.start()
        print(f"[discovery] scheduler on — daily run after {hour:02d}:00 UTC")
    else:
        print("[discovery] scheduler off (DISCOVERY_SCHEDULER=off) — use POST /discovery/run")

    return store
