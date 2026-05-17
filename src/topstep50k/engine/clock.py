"""Wall-clock abstraction with structural look-ahead protection.

The Clock is the SINGLE source of "now" inside a backtest. Components
that need historical data take a Clock and ask DataSource for "everything
up to and including now()". Any request for data strictly after now()
raises LookAheadError.

In production (live trading), a different Clock implementation backed by
the system clock can be swapped in — the strategy code is unchanged.
"""

from __future__ import annotations

from datetime import datetime, timezone


class LookAheadError(RuntimeError):
    """Raised when a component requests data strictly newer than now()."""


class Clock:
    """Monotone backtest clock. Advanced ONLY by the engine event loop.

    Strategies and indicators receive a read-only view via `now()`. The
    `advance_to` setter is intentionally not on a public Protocol — only
    the engine should import and call it.
    """

    __slots__ = ("_now",)

    def __init__(self, start: datetime) -> None:
        if start.tzinfo is None:
            raise ValueError("Clock start must be timezone-aware")
        self._now = start.astimezone(timezone.utc)

    def now(self) -> datetime:
        return self._now

    def advance_to(self, ts: datetime) -> None:
        """Engine-only. Moves the clock forward; never backward."""
        if ts.tzinfo is None:
            raise ValueError("Clock advance requires timezone-aware ts")
        ts = ts.astimezone(timezone.utc)
        if ts < self._now:
            raise LookAheadError(
                f"Clock cannot move backwards: {self._now.isoformat()} -> {ts.isoformat()}"
            )
        self._now = ts

    def assert_visible(self, ts: datetime, what: str = "data") -> None:
        """Components call this before exposing a bar to a strategy.

        Raises LookAheadError if `ts` > now(). Equal is fine: a bar that
        closed at exactly now() is observable.
        """
        ts_utc = ts.astimezone(timezone.utc) if ts.tzinfo else None
        if ts_utc is None:
            raise ValueError(f"{what} timestamp must be timezone-aware")
        if ts_utc > self._now:
            raise LookAheadError(
                f"Look-ahead: {what} at {ts_utc.isoformat()} requested while now={self._now.isoformat()}"
            )
