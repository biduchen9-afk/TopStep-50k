"""Day-0 checks for the live tick feed (PR19): run once, in Globex hours, BEFORE starting the forward test.
   python -m paper.day0 --symbol NQ.F --dataset futures [--minutes 65] [--key-file PATH]
Connects to the live tick stream, shows which fields a tick carries (so the size / volume field is identified, not assumed), builds one-minute bars from the ticks for about an hour and compares them with
the vendor's own one-minute candles for the same minutes. It writes day0_report.json (counts and pass/fail only; no key, no prices). Checks:
  1 size   a size / volume field is present (a feed with only prices is a NO-GO: the VWAP sleeves weight by volume);
  2 time   tick timestamps parse as UTC and a 'replay' flag exists;
  3 match  60 or more minutes: the tick-built close is within one tick (0.25) of the vendor candle close in at least 95% of the minutes both have;
  4 holes  minutes missing from the tick-built bars during Globex hours (the vendor candles are known to have holes, so this is reported, not decisive).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Iterable, List

from paper.tickfeed import SIZE_FIELDS


def inspect_ticks(ticks: Iterable[object]) -> dict:
    """Which fields a tick carries, from a handful of sample ticks (names and types; no values kept)."""
    seen: Dict[str, str] = {}
    n = 0
    for tk in ticks:
        n += 1
        names = [a for a in dir(tk) if not a.startswith("_") and not callable(getattr(tk, a, None))]
        for a in names:
            seen.setdefault(a, type(getattr(tk, a)).__name__)
    size_field = next((f for f in SIZE_FIELDS if f in seen and seen[f] in ("int", "float")), None)
    return dict(sampled=n, fields=seen, size_field=size_field, has_replay_flag="replay" in seen, has_timestamp="timestamp" in seen)


def compare_bars(tick_bars: List[object], candles: List[object], tol: float = 0.25) -> dict:
    """Tick-built bars against vendor candles on the minutes both have: share with the close within `tol`, and the share whose volume is within 5%."""
    c = {b.ts: b for b in candles}
    both = [(b, c[b.ts]) for b in tick_bars if b.ts in c]
    close_ok = sum(1 for a, v in both if abs(a.c - v.c) <= tol + 1e-9)
    vol_ok = sum(1 for a, v in both if v.v > 0 and abs(a.v - v.v) <= 0.05 * v.v)
    return dict(minutes_tick_bars=len(tick_bars), minutes_candles=len(candles), minutes_both=len(both), close_match=close_ok / len(both) if both else float("nan"),
                volume_match=vol_ok / len(both) if both else float("nan"), minutes_only_in_ticks=sum(1 for b in tick_bars if b.ts not in c), minutes_only_in_candles=sum(1 for t in c if t not in {b.ts for b in tick_bars}))


def verdict(info: dict, cmp_: dict, n_minutes_required: int = 60) -> dict:
    checks = {
        "1 size field present": bool(info.get("size_field")),
        "2 UTC timestamp and replay flag": bool(info.get("has_timestamp") and info.get("has_replay_flag")),
        "3 close within one tick in >= 95% of >= 60 minutes": bool(cmp_.get("minutes_both", 0) >= n_minutes_required and cmp_.get("close_match", 0) >= 0.95),
    }
    return dict(checks=checks, go=all(checks.values()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="NQ.F")
    ap.add_argument("--dataset", default="futures")
    ap.add_argument("--minutes", type=int, default=65, help="bars to collect from the live stream (default 65)")
    ap.add_argument("--key-file", default=None)
    ap.add_argument("--out", default="day0_report.json")
    a = ap.parse_args(argv)
    from paper.run import load_key_file
    load_key_file(a.key_file)
    key = os.environ.get("LSE_API_KEY")
    if not key:
        sys.exit("LSE_API_KEY is not set (environment variable or --key-file)")
    import itertools
    import pandas as pd
    from paper.feed import lse_fetch
    from paper.tickfeed import TickBars, sdk_ticks
    if not TickBars.market_open(pd.Timestamp.now(tz="UTC")):
        sys.exit("The market is closed now (Globex: Sunday 18:00 to Friday 17:00 ET, daily halt 17:00-18:00). Run this during trading hours.")
    sample: List[object] = []

    def connect(start):
        for tk in sdk_ticks(a.symbol, key, start):
            if len(sample) < 200:
                sample.append(tk)
            yield tk
    tb = TickBars(connect, start=None, settle=3.0, stall=90.0)
    t0 = pd.Timestamp.now(tz="UTC").ceil("min")
    print(f"collecting {a.minutes} one-minute bars from the live stream (about {a.minutes} minutes)...", flush=True)
    bars = [b for b in itertools.islice(tb, a.minutes)]
    time.sleep(150)                                                   # let the vendor finish its own candles for the last minutes
    start = t0.strftime("%Y-%m-%dT%H:%M:%S"); end = (bars[-1].ts.tz_convert("UTC") + pd.Timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%S") if bars else start
    candles = lse_fetch(a.symbol, start, end=end, dataset=a.dataset)
    info, cmp_ = inspect_ticks(sample), compare_bars(bars, candles)
    v = verdict(info, cmp_)
    rep = dict(symbol=a.symbol, dataset=a.dataset, tick_fields=info["fields"], size_field=info["size_field"], volume_basis=tb.volume_basis, comparison=cmp_, reconnects=tb.reconnects, rejected=dict(future=tb.n_future, price=tb.n_bad_price, size=tb.n_bad_size), **v)
    json.dump(rep, open(a.out, "w"), indent=1, default=str)
    print(json.dumps(rep, indent=1, default=str))
    print("\nDAY-0:", "GO" if v["go"] else "NO-GO", "| failed:", [k for k, ok in v["checks"].items() if not ok] or "none")
    return 0 if v["go"] else 1


if __name__ == "__main__":
    sys.exit(main())
