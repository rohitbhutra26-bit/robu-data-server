#!/usr/bin/env python3
"""
Robu data-quality scanner.

Sweeps a list of stocks and flags systematic data/metric problems so they can be fixed at
the ROOT (one fix → whole class), instead of hunting stock-by-stock.

Usage:
  python3 scripts/data_quality_scan.py                 # scan the built-in diverse sample
  python3 scripts/data_quality_scan.py SYM1 SYM2 ...    # scan specific symbols
  python3 scripts/data_quality_scan.py --file syms.txt  # scan symbols from a file
  BASE=http://localhost:8000 python3 scripts/data_quality_scan.py

Exit code is non-zero if any CRITICAL class is found (handy for the daily health check).
"""
import os, sys, json, urllib.request, concurrent.futures
from collections import defaultdict

BASE = os.environ.get("BASE", "https://robu-data-server-production.up.railway.app")

# Sector labels Robu maps to a real model. Keep in sync with sectorModelMap.ts (or fetch it).
# A label NOT here falls through to the 25x Broad-Market P/E default → likely mis-valuation.
# (This list is intentionally permissive; the scan flags genuinely-unmapped tails.)
def _fetch(path):
    try:
        with urllib.request.urlopen(f"{BASE}{path}", timeout=15) as r:
            return json.load(r)
    except Exception:
        return None

SAMPLE = ("RELIANCE TCS INFY HDFCBANK ICICIBANK SBIN ITC LT KOTAKBANK AXISBANK BAJFINANCE "
          "SUNPHARMA MARUTI TITAN PERSISTENT KAYNES BROOKS RATEGAIN PFC RECLTD IRFC AUBANK "
          "BHARATFORG CYIENT DELHIVERY NTPC POWERGRID GAIL TATASTEEL EMBASSY SBILIFE TMPV "
          "JIOFIN SWIGGY OLAELEC HYUNDAI ZOMATO PAYTM DIVISLAB BIOCON DMART TRENT").split()

def scan(symbols):
    flags = defaultdict(list)
    def one(sym):
        d = _fetch(f"/company-v2/{sym}")
        if not d: return ('FETCH_FAIL', sym, None)
        return (None, sym, d)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(one, symbols))
    n = 0
    for err, sym, d in results:
        if err: flags['FETCH_FAIL'].append(sym); continue
        n += 1
        price=d.get('currentPrice') or 0; pe=d.get('pe') or 0; pb=d.get('pb') or 0
        eps=d.get('eps') or 0; book=d.get('bookValue') or 0; roe=d.get('roe') or 0
        sec=(d.get('sector') or '').strip()
        if price<=0: flags['CRITICAL:NO_PRICE'].append(sym)
        if not sec or sec in ('Unknown','NSE Listed'): flags['NO_SECTOR'].append(sym)
        if 0<pe<4: flags['DISTORTED_LOW_PE'].append(f'{sym}(pe{pe})')
        if pe>200: flags['EXTREME_HIGH_PE'].append(f'{sym}(pe{pe})')
        if eps<0 or roe<0: flags['LOSS_OR_NEG_ROE'].append(sym)
        if book<=0 and pb>0: flags['MISSING_BOOK'].append(sym)
        if eps>0 and pe>3 and price>0 and abs(pe*eps-price)/price>0.25:
            flags['PE_RECONCILE_OFF'].append(f'{sym}(pe*eps={pe*eps:.0f}/px{price:.0f})')
        if pb>0 and book>0 and price>0 and abs(pb*book-price)/price>0.25:
            flags['PB_RECONCILE_OFF'].append(f'{sym}(pb*bv={pb*book:.0f}/px{price:.0f})')
    return n, flags

if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--rotate":
        # daily rotating batch: a different slice of the universe each day → full coverage over time
        import datetime
        n = int(args[1]) if len(args) > 1 else 100
        off = (datetime.date.today().toordinal() * n)
        u = _fetch(f"/universe/symbols?offset={off}&limit={n}") or {}
        syms = u.get("symbols", [])
        print(f"rotating batch: offset {u.get('offset')}/{u.get('total')} ({len(syms)} symbols)")
    elif args and args[0] == "--file":
        syms = [l.strip() for l in open(args[1]) if l.strip()]
    elif args:
        syms = args
    else:
        syms = SAMPLE
    n, flags = scan(syms)
    print(f"\n=== Robu data-quality scan — {n}/{len(syms)} stocks · {BASE} ===\n")
    if not flags:
        print("  clean — no issues found."); sys.exit(0)
    for k in sorted(flags, key=lambda x: -len(flags[x])):
        ex = ', '.join(flags[k][:8])
        print(f"  {len(flags[k]):>3}  {k:<22} e.g. {ex}")
    critical = any(k.startswith('CRITICAL') for k in flags)
    sys.exit(1 if critical else 0)
