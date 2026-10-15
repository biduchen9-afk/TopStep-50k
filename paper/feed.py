"""Minute-bar feeds. Every feed yields `Bar` objects (completed one-minute bars, bar-OPEN timestamps in America/New_York) in time order.

  ReplayFeed   stored Databento sessions (market_data.Sessions) replayed as if live - used for parity tests, demos and as a stand-in while the live feed is not available
  LSEBars      London Strategic Edge vault REST API (https://api.londonstrategicedge.com/vault/candles), polled; the API key is read from the LSE_API_KEY environment variable ONLY
               (never written to a file); stdlib urllib client written from the SDK's public behaviour (x-api-key header, /candles?symbol&timeframe&start&end&limit&order)
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Iterator, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")
UTC = timezone.utc
VAULT_URL = "https://api.londonstrategicedge.com/vault"
USER_AGENT = "topstep-paper-harness (+python-urllib)"


@dataclass
class Bar:
    ts: pd.Timestamp        # bar-open time, tz-aware America/New_York
    o: float
    h: float
    l: float
    c: float
    v: float

    @property
    def smin(self) -> int:
        """Minutes since the 18:00 ET session open (0..1439)."""
        return (self.ts.hour * 60 + self.ts.minute - 18 * 60) % 1440

    @property
    def label(self):
        """Session label: the calendar date of the 16:00 ET end (Sunday evening belongs to Monday)."""
        return (self.ts.tz_localize(None) + pd.Timedelta(hours=6)).normalize().date()


def in_session(bar: Bar) -> bool:
    """18:00 -> 16:00 ET on weekday labels (the 16:00-18:00 window and weekends are not traded)."""
    return bar.smin < 1320 and bar.label.weekday() < 5


class ReplayFeed:
    """Replay of stored sessions from `from_label` (a date) to `to_label`, yielding bars in order. Optional `speed` = bars per second (None = as fast as possible)."""

    def __init__(self, sess, from_label, to_label=None, speed: Optional[float] = None):
        self.sess, self.speed = sess, speed
        labels = pd.DatetimeIndex(sess.dates)
        self.i0 = int(np.searchsorted(labels.values, np.datetime64(pd.Timestamp(from_label).date())))
        self.i1 = len(sess) if to_label is None else int(np.searchsorted(labels.values, np.datetime64(pd.Timestamp(to_label).date()), side="right"))

    def __iter__(self) -> Iterator[Bar]:
        for i in range(self.i0, self.i1):
            o, h, l, c, v, m = self.sess.day(i)
            label = pd.Timestamp(self.sess.dates[i])
            start = (label - pd.Timedelta(days=1)).replace(hour=18, minute=0)          # wall clock of the 18:00 ET open (Sunday for a Monday label)
            for j in range(len(c)):
                ts = (start + pd.Timedelta(minutes=int(m[j]))).tz_localize(ET)
                yield Bar(ts, float(o[j]), float(h[j]), float(l[j]), float(c[j]), float(v[j]))
                if self.speed:
                    time.sleep(1.0 / self.speed)


# ------------------------------------------------------------------------------------------------ London Strategic Edge
class LSEError(RuntimeError):
    pass


def _lse_get(path: str, params: Optional[List[tuple]] = None, key: Optional[str] = None, timeout: float = 60.0):
    key = key or os.environ.get("LSE_API_KEY")
    if not key:
        raise LSEError("set the LSE_API_KEY environment variable (the key is never stored by this code)")
    qs = urllib.parse.urlencode([(k, str(v)) for k, v in (params or []) if v is not None])
    req = urllib.request.Request(f"{VAULT_URL}{path}" + (f"?{qs}" if qs else ""), headers={"x-api-key": key, "User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise LSEError(f"[{e.code}] {e.read().decode('utf-8', 'replace')[:300]}")
    except OSError as e:
        raise LSEError(f"request failed before an HTTP response: {getattr(e, 'reason', None) or e}")


def lse_catalog(key: Optional[str] = None) -> List[dict]:
    """Every instrument in the vault: rows {dataset, symbol, ticks, first_tick, last_tick, ...}."""
    return _lse_get("/catalog", key=key)


def lse_find(pattern: str, key: Optional[str] = None, catalog: Optional[List[dict]] = None) -> List[dict]:
    cat = catalog if catalog is not None else lse_catalog(key)
    p = pattern.upper()
    return [r for r in cat if p in str(r.get("symbol", "")).upper() or p in str(r.get("name", "")).upper()]


def _to_et(ts) -> pd.Timestamp:
    t = pd.Timestamp(str(ts).replace(" ", "T") if " " in str(ts) and "T" not in str(ts) else ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert(ET)


def parse_candles(rows: List[dict]) -> List[Bar]:
    """Vault candle rows -> Bars. Accepts ts/timestamp/time and open/high/low/close/volume (or o/h/l/c/v); timestamps are UTC bar-open times."""
    bars = []
    for r in rows:
        g = lambda *ks: next((r[k] for k in ks if k in r and r[k] is not None), None)
        t = g("ts", "timestamp", "time", "t")
        if t is None:
            continue
        o, h, l, c = g("open", "o"), g("high", "h"), g("low", "l"), g("close", "c")
        if None in (o, h, l, c):
            continue
        bars.append(Bar(_to_et(t), float(o), float(h), float(l), float(c), float(g("volume", "v") or 0.0)))
    bars.sort(key=lambda b: b.ts)
    return bars


def lse_fetch(symbol: str, start: str, end: Optional[str] = None, timeframe: str = "1m", key: Optional[str] = None, dataset: Optional[str] = None, pause: float = 0.4,
              limit: int = 5000, max_calls: int = 10000) -> List[Bar]:
    """Page through /candles (at most 5000 rows per call) from `start` (UTC ISO) to `end`."""
    out: List[Bar] = []
    cur = start
    for _ in range(max_calls):
        rows = _lse_get("/candles", [("symbol", symbol), ("timeframe", timeframe), ("order", "asc"), ("limit", limit), ("dataset", dataset), ("start", cur), ("end", end)], key=key)
        if not rows:
            break
        bars = parse_candles(rows)
        if not bars:
            break
        if out and bars[-1].ts <= out[-1].ts:
            break
        out.extend(b for b in bars if not out or b.ts > out[-1].ts)
        last_utc = bars[-1].ts.tz_convert("UTC")
        nxt = (last_utc + pd.Timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%S")
        if nxt == cur or len(rows) < limit:
            break
        cur = nxt
        time.sleep(pause)
    return out


class LSEBars:
    """Live feed: every `poll` seconds fetch the bars since the last one and yield those that are complete (bar open + 1 minute + `settle` seconds in the past)."""

    def __init__(self, symbol: str, start: str, timeframe: str = "1m", key: Optional[str] = None, dataset: Optional[str] = None, poll: float = 20.0, settle: float = 5.0, stop_after_idle: Optional[float] = None):
        self.symbol, self.start, self.timeframe, self.key, self.dataset = symbol, start, timeframe, key, dataset
        self.poll, self.settle, self.stop_after_idle = poll, settle, stop_after_idle

    def __iter__(self) -> Iterator[Bar]:
        cur = self.start
        last: Optional[pd.Timestamp] = None
        idle_since = time.time()
        while True:
            got = lse_fetch(self.symbol, cur, None, self.timeframe, self.key, self.dataset)
            now = pd.Timestamp.now(tz="UTC")
            n_new = 0
            for b in got:
                if last is not None and b.ts <= last:
                    continue
                if b.ts.tz_convert("UTC") + pd.Timedelta(minutes=1, seconds=self.settle) > now:
                    break
                last = b.ts
                n_new += 1
                yield b
            if last is not None:
                cur = (last.tz_convert("UTC") + pd.Timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%S")
            if n_new:
                idle_since = time.time()
            elif self.stop_after_idle is not None and time.time() - idle_since > self.stop_after_idle:
                return
            time.sleep(self.poll)
