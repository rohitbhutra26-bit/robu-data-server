"""Discovery Store — SQLite persistence.

Why SQLite over a JSON file: atomic writes (no torn reads while the nightly job
writes), concurrent reads from FastAPI, and built-in history for backtesting.

Two tables:
  runs(id, run_ts, status, count, source)
  records(run_id, symbol, score, category, payload_json, price_at_run)
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import sqlite3
import json
import os
import threading
from datetime import datetime, timezone

_LOCK = threading.Lock()


class DiscoveryStore:
    def __init__(self, data_dir: str):
        os.makedirs(data_dir, exist_ok=True)
        self.path = os.path.join(data_dir, "discovery.db")
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, timeout=30)
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self) -> None:
        with _LOCK, self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_ts TEXT NOT NULL,
                status TEXT NOT NULL,
                count INTEGER DEFAULT 0,
                source TEXT DEFAULT 'unknown'
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS records (
                run_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                score INTEGER NOT NULL,
                category TEXT,
                price_at_run REAL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (run_id, symbol)
            )""")
            c.execute("CREATE INDEX IF NOT EXISTS idx_records_symbol ON records(symbol)")

    # ── write ────────────────────────────────────────────────────────────
    def start_run(self) -> int:
        with _LOCK, self._conn() as c:
            cur = c.execute(
                "INSERT INTO runs (run_ts, status) VALUES (?, 'running')",
                (datetime.now(timezone.utc).isoformat(),),
            )
            return cur.lastrowid

    def finish_run(self, run_id: int, records: List[Dict[str, Any]],
                   prices: Dict[str, float], source: str) -> None:
        with _LOCK, self._conn() as c:
            for r in records:
                c.execute(
                    "INSERT OR REPLACE INTO records (run_id, symbol, score, category, price_at_run, payload_json) "
                    "VALUES (?,?,?,?,?,?)",
                    (run_id, r["symbol"], r["discoveryScore"], r["category"],
                     prices.get(r["symbol"]), json.dumps(r)),
                )
            c.execute("UPDATE runs SET status='done', count=?, source=? WHERE id=?",
                      (len(records), source, run_id))

    def fail_run(self, run_id: int, msg: str) -> None:
        with _LOCK, self._conn() as c:
            c.execute("UPDATE runs SET status=? WHERE id=?", (f"error: {msg[:120]}", run_id))

    # ── read ─────────────────────────────────────────────────────────────
    def latest_run(self) -> Optional[sqlite3.Row]:
        with self._conn() as c:
            return c.execute(
                "SELECT * FROM runs WHERE status='done' ORDER BY id DESC LIMIT 1"
            ).fetchone()

    def latest_records(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        run = self.latest_run()
        if not run:
            return []
        with self._conn() as c:
            rows = c.execute(
                "SELECT payload_json FROM records WHERE run_id=? ORDER BY score DESC",
                (run["id"],),
            ).fetchall()
        records = [json.loads(r["payload_json"]) for r in rows]
        if category and category != "All":
            records = [r for r in records if r.get("category") == category]
        return records

    def status(self) -> Dict[str, Any]:
        with self._conn() as c:
            last = c.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 1").fetchone()
            total_runs = c.execute("SELECT COUNT(*) AS n FROM runs WHERE status='done'").fetchone()["n"]
        return {
            "lastRun": dict(last) if last else None,
            "completedRuns": total_runs,
        }

    def previous_symbols(self) -> set:
        """Symbols from the prior done-run (to flag what's genuinely 'new')."""
        with self._conn() as c:
            runs = c.execute(
                "SELECT id FROM runs WHERE status='done' ORDER BY id DESC LIMIT 2"
            ).fetchall()
            if len(runs) < 2:
                return set()
            prev_id = runs[1]["id"]
            rows = c.execute("SELECT symbol FROM records WHERE run_id=?", (prev_id,)).fetchall()
            return {r["symbol"] for r in rows}

    def history_for(self, symbol: str) -> List[Dict[str, Any]]:
        """Score + price over time for one symbol (used by backtest)."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT r.run_ts, rec.score, rec.price_at_run "
                "FROM records rec JOIN runs r ON r.id = rec.run_id "
                "WHERE rec.symbol=? AND r.status='done' ORDER BY r.id ASC",
                (symbol,),
            ).fetchall()
        return [dict(x) for x in rows]
