"""One-minute bars built from the London Strategic Edge live TICK stream (WebSocket), so the harness does not depend on the vault's candles (which had holes, FINDINGS section 32).

   from paper.tickfeed import TickBars, sdk_ticks
   feed = TickBars(lambda start: sdk_ticks("NQ.F", key, start), start="2026-10-15T09:00:00")

`sdk_ticks` uses the vendor package (`pip install lse-data`; `LSE(api_key).stream([symbol], start=...)` replays up to 24 hours of missed ticks, then continues live; every tick has `.symbol`, `.price`, `.timestamp`, `.replay`).
A tick may also carry a size / volume field (`size`, `volume`, `qty`, `quantity`, `last_size`): the first one found is used; without any, the number of ticks stands in for volume (reported by `volume_basis`).
`TickBars` is a plain iterator of `paper.feed.Bar`. A minute is emitted when a tick of a later minute arrives, or, live, when no tick has arrived for `settle` seconds after the minute ended. When the stream drops it reconnects with
`start` = the beginning of the unfinished minute (the partial minute is rebuilt from the replay, so nothing is counted twice) and bars at or before the last emitted minute are never emitted again.
"""
from __future__ import annotations

import math
import queue
import threading
import time
from typing import Callable, Iterator, Optional

import pandas as pd

from paper.feed import Bar, ET

SIZE_FIELDS = ("size", "volume", "qty", "quantity", "last_size", "amount")


def _ts_utc(x) -> pd.Timestamp:
    """Tick timestamp -> tz-aware UTC pandas Timestamp (accepts datetime / ISO string / epoch s, ms, us, ns; naive values are taken as UTC)."""
    if isinstance(x, (int, float)):
        v = float(x)
        unit = "ns" if v > 1e17 else "us" if v > 1e14 else "ms" if v > 1e11 else "s"
        return pd.Timestamp(v, unit=unit, tz="UTC")
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def sdk_ticks(symbol: str, key: str, start: Optional[str] = None) -> Iterator[object]:
    try:
        from lse import LSE
    except ImportError as e:                                      # pragma: no cover
        raise SystemExit("the live tick feed needs the vendor package: pip install lse-data") from e
    client = LSE(api_key=key)
    yield from (client.stream([symbol], start=start) if start else client.stream([symbol]))


class TickBars:
    def __init__(self, connect: Callable[[Optional[str]], Iterator[object]], start: Optional[str] = None, settle: float = 3.0, backoff: float = 5.0, max_idle: Optional[float] = None,
                 clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep, max_reconnects: Optional[int] = None,
                 stall: Optional[float] = None, max_replay_hours: Optional[float] = None, max_future: float = 90.0):
        self.connect, self.start, self.settle, self.backoff, self.max_idle = connect, start, settle, backoff, max_idle
        self.clock, self.sleep, self.max_reconnects = clock, sleep, max_reconnects
        self.volume_basis = "ticks (no size field seen)"
        self.n_ticks = 0
        self.last_emitted: Optional[pd.Timestamp] = None           # UTC start of the last emitted minute
        self.reconnects = 0
        self.stall, self.max_replay_hours, self.max_future = stall, max_replay_hours, max_future
        self.last_error: Optional[str] = None
        self.n_future = self.n_bad_price = self.n_bad_size = self.n_stall = 0
        self._fail_streak = 0; self._fut_run = 0

    @staticmethod
    def market_open(ts_utc: pd.Timestamp) -> bool:
        """CME equity-index Globex hours in ET: Sunday 18:00 to Friday 17:00, daily halt 17:00-18:00 (holidays not modelled)."""
        t = ts_utc.tz_convert(ET); m, wd = t.hour * 60 + t.minute, t.weekday()
        return not (wd == 5 or (wd == 4 and m >= 1020) or (wd == 6 and m < 1080) or 1020 <= m < 1080)

    # ---------------------------------------------------------------- tick -> (minute start UTC, price, size)
    def _parse(self, tick):
        price = float(getattr(tick, "price"))
        if not (math.isfinite(price) and price > 0):
            self.n_bad_price += 1
            raise ValueError("bad price")
        size = None
        for f in SIZE_FIELDS:
            v = getattr(tick, f, None)
            if v is not None:
                try:
                    size = float(v); self.volume_basis = f; break
                except (TypeError, ValueError):
                    pass
        if size is not None and not (math.isfinite(size) and size >= 0):
            self.n_bad_size += 1; size = 0.0                           # NaN / negative size: count the tick, not the contracts
        t = _ts_utc(getattr(tick, "timestamp"))
        return t.floor("min"), price, (size if size is not None else 1.0), bool(getattr(tick, "replay", False))

    @staticmethod
    def _bar(minute_utc, o, h, l, c, v) -> Bar:
        return Bar(minute_utc.tz_convert(ET), o, h, l, c, v)

    def __iter__(self) -> Iterator[Bar]:
        cur = None                                                 # [minute_utc, o, h, l, c, v]
        while True:
            q: "queue.Queue" = queue.Queue(maxsize=100000)
            stop = threading.Event()

            def pump(src_start):
                try:
                    for tk in self.connect(src_start):
                        if stop.is_set():
                            return
                        q.put(("tick", tk))
                    q.put(("end", None))
                except Exception as e:                              # the stream dropped
                    self.last_error = f"{type(e).__name__}: {e}"[:200]
                    q.put(("error", e))
            src_start = cur_start = (cur[0].tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S") if cur is not None else
                                     (self.last_emitted + pd.Timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%S") if self.last_emitted is not None else self.start)
            if self.max_replay_hours is not None and src_start is not None:    # the vendor replays a limited window: never ask for an older start (a long outage becomes a hole, not a retry loop)
                floor_ = (pd.Timestamp(self.clock(), unit="s", tz="UTC") - pd.Timedelta(hours=self.max_replay_hours)).ceil("min")
                if pd.Timestamp(src_start, tz="UTC") < floor_:
                    src_start = floor_.strftime("%Y-%m-%dT%H:%M:%S")
            cur = None                                              # an unfinished minute is rebuilt from the replay
            th = threading.Thread(target=pump, args=(src_start,), daemon=True); th.start()
            last_tick_wall = self.clock(); ended = False; last_replay = False; kind = tk = None
            while True:
                try:
                    kind, tk = q.get(timeout=0.5)
                except queue.Empty:
                    now = self.clock()
                    if self.stall and now - last_tick_wall > self.stall and self.market_open(pd.Timestamp(now, unit="s", tz="UTC")):
                        self.n_stall += 1; self.last_error = f"no tick for {now - last_tick_wall:.0f} s while the market is open: reconnecting"
                        ended = True
                        break
                    # wall-clock flush only for a live minute that ended minutes ago at most (never while ticks are being replayed: a replay pause must not cut a minute)
                    if cur is not None and not last_replay and now - last_tick_wall >= self.settle and cur[0].timestamp() + 60 + self.settle <= now <= cur[0].timestamp() + 60 + 300:
                        b = self._flush(cur); cur = None
                        if b is not None:
                            yield b
                    if self.max_idle is not None and now - last_tick_wall > self.max_idle:
                        stop.set(); return
                    continue
                last_tick_wall = self.clock()
                if kind in ("end", "error"):
                    ended = True
                    break
                self.n_ticks += 1; self._fail_streak = 0
                try:
                    minute, price, size, replay = self._parse(tk)
                except Exception:
                    continue
                last_replay = replay
                if not replay and minute > pd.Timestamp(self.clock(), unit="s", tz="UTC").floor("min") + pd.Timedelta(seconds=self.max_future):
                    self.n_future += 1; self._fut_run += 1          # a tick stamped in the future would black out the real ones until its time comes
                    if self._fut_run <= 30:
                        continue                                    # one bad stamp is dropped; 30 in a row means the machine clock is slow: accept and say so
                    self.last_error = "tick stamps run ahead of the machine clock: check the clock (NTP)"
                else:
                    self._fut_run = 0
                if self.last_emitted is not None and minute <= self.last_emitted:
                    continue                                        # a minute already emitted (reconnect replay)
                if cur is None:
                    cur = [minute, price, price, price, price, size]
                elif minute == cur[0]:
                    cur[2] = max(cur[2], price); cur[3] = min(cur[3], price); cur[4] = price; cur[5] += size
                elif minute > cur[0]:
                    b = self._flush(cur)
                    if b is not None:
                        yield b
                    cur = [minute, price, price, price, price, size]
                # a tick older than the current minute (out of order) is ignored
            stop.set()
            if not ended:
                return
            if kind == "error":
                self.last_error = f"{type(tk).__name__}: {tk}"[:200]
            self._fail_streak += 1
            # the stream dropped or ended: reconnect (live) with the unfinished minute rebuilt from the replay
            self.reconnects += 1
            if self.max_reconnects is not None and self.reconnects > self.max_reconnects:
                if cur is not None:
                    b = self._flush(cur)
                    if b is not None:
                        yield b
                return
            self.sleep(min(self.backoff * 2 ** min(self._fail_streak - 1, 6), 300.0))      # exponential, capped at 5 minutes

    def _flush(self, cur) -> Optional[Bar]:
        minute = cur[0]
        if self.last_emitted is not None and minute <= self.last_emitted:
            return None
        self.last_emitted = minute
        return self._bar(minute, cur[1], cur[2], cur[3], cur[4], cur[5])
