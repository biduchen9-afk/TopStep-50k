"""Point-in-time data access.

A DataSource wraps a chronologically-ordered series of Bars per symbol
and only ever yields bars that are <= clock.now(). The engine drives
iteration via `stream()`, which is a generator that advances the clock
ONE bar at a time and emits the bar AFTER advancing.

Any code path that pulls historical context (e.g. an indicator computing
a rolling mean) goes through `history()` which calls
`clock.assert_visible()` on each bar it returns.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Iterable, Protocol

from topstep50k.engine.clock import Clock
from topstep50k.engine.types import Bar


class DataSource(Protocol):
    def symbols(self) -> list[str]: ...
    def history(self, symbol: str, n: int) -> list[Bar]:
        """The most recent n bars at or before clock.now(). Newest last."""
        ...


class BarStream(Protocol):
    def stream(self, clock: Clock) -> Iterator[tuple[str, Bar]]:
        """Yield (symbol, bar) tuples in strict chronological order,
        advancing the clock to bar.ts before yielding."""
        ...


class InMemoryBarSource:
    """All bars for all symbols held in memory.

    Suitable for daily/hourly bars across a few years. For 1-minute
    tick-level multi-year datasets, write a different DataSource backed
    by Parquet + memory-mapped access — same interface, different storage.
    """

    def __init__(self, bars_by_symbol: dict[str, list[Bar]], clock: Clock) -> None:
        # Defensive: sort and freeze
        self._bars: dict[str, list[Bar]] = {
            s: sorted(bars, key=lambda b: b.ts) for s, bars in bars_by_symbol.items()
        }
        self._clock = clock
        # _next[symbol] = index of the next bar not yet consumed
        self._next: dict[str, int] = {s: 0 for s in self._bars}

    def symbols(self) -> list[str]:
        return sorted(self._bars.keys())

    def history(self, symbol: str, n: int) -> list[Bar]:
        bars = self._bars[symbol]
        now = self._clock.now()
        # Largest index i such that bars[i].ts <= now
        # Linear is fine; switch to bisect if hot.
        idx = -1
        for i, b in enumerate(bars):
            if b.ts > now:
                break
            idx = i
        if idx < 0:
            return []
        lo = max(0, idx - n + 1)
        out = bars[lo : idx + 1]
        for b in out:
            self._clock.assert_visible(b.ts, f"history({symbol})")
        return out

    def stream(self, clock: Clock) -> Iterator[tuple[str, Bar]]:
        """Merge all symbols by timestamp and yield each bar exactly once,
        advancing the clock to bar.ts BEFORE yielding so that any
        strategy code that runs during the yield sees the right `now()`.
        """
        if clock is not self._clock:
            raise ValueError("DataSource was built with a different Clock instance")
        # Merge of pre-sorted lists; simple n-way scan since we already
        # hold everything in memory.
        cursors = {s: 0 for s in self._bars}
        while True:
            best_symbol: str | None = None
            best_ts = None
            for s, i in cursors.items():
                if i >= len(self._bars[s]):
                    continue
                ts = self._bars[s][i].ts
                if best_ts is None or ts < best_ts:
                    best_symbol = s
                    best_ts = ts
            if best_symbol is None:
                return
            bar = self._bars[best_symbol][cursors[best_symbol]]
            cursors[best_symbol] += 1
            clock.advance_to(bar.ts)
            yield best_symbol, bar


def in_memory_from_iter(
    bars_by_symbol: Iterable[tuple[str, Bar]], clock: Clock
) -> InMemoryBarSource:
    buckets: dict[str, list[Bar]] = {}
    for s, b in bars_by_symbol:
        buckets.setdefault(s, []).append(b)
    return InMemoryBarSource(buckets, clock)
