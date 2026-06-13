"""Phase 5 — Backtest / trust.

Once there are >=2 historical runs, we can ask the only question that matters:
"did high Discovery Scores actually precede better price moves?"

For each symbol that appears in an older run, compare its score then to the
forward return (price at a later run / price then). Report the average forward
return by score bucket and a simple rank correlation. This is intentionally
lightweight — it grows more meaningful as run history accumulates.
"""

from __future__ import annotations
from typing import Dict, Any, List
from .store import DiscoveryStore


def _spearman(pairs: List[tuple]) -> float:
    """Rank correlation between score and forward return. 0 if not enough data."""
    n = len(pairs)
    if n < 4:
        return 0.0

    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        for rank, i in enumerate(order):
            r[i] = rank + 1
        return r

    scores = [p[0] for p in pairs]
    rets = [p[1] for p in pairs]
    rs, rr = ranks(scores), ranks(rets)
    d2 = sum((rs[i] - rr[i]) ** 2 for i in range(n))
    return round(1 - (6 * d2) / (n * (n * n - 1)), 3)


def run_backtest(store: DiscoveryStore) -> Dict[str, Any]:
    status = store.status()
    if status["completedRuns"] < 2:
        return {
            "ready": False,
            "message": "Need at least 2 completed runs to backtest. Scores are being recorded; check back after the next run.",
            "completedRuns": status["completedRuns"],
        }

    # Build (score_then, forward_return) pairs across each symbol's history.
    latest = store.latest_records()
    symbols = {r["symbol"] for r in latest}
    pairs: List[tuple] = []
    buckets = {"high (75+)": [], "mid (60-74)": [], "low (<60)": []}

    for sym in symbols:
        hist = store.history_for(sym)
        usable = [h for h in hist if h.get("price_at_run")]
        if len(usable) < 2:
            continue
        first, last = usable[0], usable[-1]
        if not first["price_at_run"]:
            continue
        fwd_ret = (last["price_at_run"] - first["price_at_run"]) / first["price_at_run"] * 100
        score_then = first["score"]
        pairs.append((score_then, fwd_ret))
        if score_then >= 75:
            buckets["high (75+)"].append(fwd_ret)
        elif score_then >= 60:
            buckets["mid (60-74)"].append(fwd_ret)
        else:
            buckets["low (<60)"].append(fwd_ret)

    def avg(xs):
        return round(sum(xs) / len(xs), 2) if xs else None

    return {
        "ready": True,
        "completedRuns": status["completedRuns"],
        "sampleSize": len(pairs),
        "rankCorrelation": _spearman(pairs),
        "avgForwardReturnByBucket": {k: {"avgReturnPct": avg(v), "n": len(v)} for k, v in buckets.items()},
        "note": "Forward return = price change between a symbol's first and latest recorded run. More runs = more reliable.",
    }
