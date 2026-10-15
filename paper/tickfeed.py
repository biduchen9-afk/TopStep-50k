"""One-minute bars built from the London Strategic Edge live TICK stream (WebSocket), so the harness does not depend on the vault's candles (which had holes, FINDINGS section 32).

   from paper.tickfeed import TickBars, sdk_ticks
   feed = TickBars(lambda start: sdk_ticks("NQ.F", key, start), start="2026-10-15T09:00:00")

`sdk_ticks` uses the vendor package (`pip install lse-data`; `LSE(api_key).stream([symbol], start=...)` replays up to 24 hours of missed ticks, then continues live; every tick has `.symbol`, `.price`, `.timestamp`, `.replay`).
A tick may also carry a size / volume field (`size`, `volume`, `qty`, `quantity`, `last_size`): the first one found is used; without any, the number of ticks stands in for volume (reported by `volume_basis`).
`TickBars` is a plain iterator of `paper.feed.Bar`. A minute is emitted when a tick of a later minute arrives, or, live, when no tick has arrived for `settle` seconds after the minute ended. When the stream drops it reconnects with
`start` = the beginning of the unfinished minute (the partial minute is rebuilt from the replay, so nothing is counted twice) and bars at or before the last emitted minute are never emitted again.
"""
from __future__ import annotations

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
                 clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep, max_reconnects: Optional[int] = None):
        self.connect, self.start, self.settle, self.backoff, self.max_idle = connect, start, settle, backoff, max_idle
        self.clock, self.sleep, self.max_reconnects = clock, sleep, max_reconnects
        self.volume_basis = "ticks (no size field seen)"
        self.n_ticks = 0
        self.last_emitted: Optional[pd.Timestamp] = None           # UTC start of the last emitted minute
        self.reconnects = 0

    # ---------------------------------------------------------------- tick -> (minute start UTC, price, size)
    def _parse(self, tick):
        price = float(getattr(tick, "price"))
        size = None
        for f in SIZE_FIELDS:
            v = getattr(tick, f, None)
            if v is not None:
                try:
                    size = float(v); self.volume_basis = f; break
                except (TypeError, ValueError):
                    pass
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
                    q.put(("error", e))
            src_start = cur_start = (cur[0].tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S") if cur is not None else
                                     (self.last_emitted + pd.Timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%S") if self.last_emitted is not None else self.start)
            cur = None                                              # an unfinished minute is rebuilt from the replay
            th = threading.Thread(target=pump, args=(src_start,), daemon=True); th.start()
            last_tick_wall = self.clock(); ended = False
            while True:
                try:
                    kind, tk = q.get(timeout=0.5)
                except queue.Empty:
                    now = self.clock()
                    if cur is not None and now - last_tick_wall >= self.settle and now >= cur[0].timestamp() + 60 + self.settle:
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
                self.n_ticks += 1
                try:
                    minute, price, size, replay = self._parse(tk)
                except Exception:
                    continue
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
            # the stream dropped or ended: reconnect (live) with the unfinished minute rebuilt from the replay
            self.reconnects += 1
            if self.max_reconnects is not None and self.reconnects > self.max_reconnects:
                if cur is not None:
                    b = self._flush(cur)
                    if b is not None:
                        yield b
                return
            self.sleep(self.backoff)

    def _flush(self, cur) -> Optional[Bar]:
        minute = cur[0]
        if self.last_emitted is not None and minute <= self.last_emitted:
            return None
        self.last_emitted = minute
        return self._bar(minute, cur[1], cur[2], cur[3], cur[4], cur[5])
