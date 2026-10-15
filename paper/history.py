"""Per-session history needed to build the context of the NEXT session without look-ahead: trailing ATR (mean of the previous 14 non-roll session ranges), the previous close and the
time-of-day noise-area sigmas (mean over the previous 14 sessions of |price at t / open - 1|). Mirrors academic_strategies.context and intraday_strategies.trailing_atr exactly."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

CASH_OPEN = 15 * 60 + 30
STEP = 30
K_DEC = 12
ATR_LOOKBACK = 14


def session_move(o, c, smin) -> np.ndarray:
    """|close of the last bar before t / open at 09:30 - 1| for the twelve decision times t = 10:00 ... 15:30 (NaN when not available), as academic_strategies.context."""
    mv = np.full(K_DEC, np.nan)
    a = int(np.searchsorted(smin, CASH_OPEN))
    if a >= len(c) - 5:
        return mv
    op = o[a]
    for k in range(1, K_DEC + 1):
        idx = int(np.searchsorted(smin, CASH_OPEN + STEP * k))
        if idx > a:
            mv[k - 1] = abs(c[idx - 1] / op - 1.0)
    return mv


@dataclass
class History:
    ranges: List[float] = field(default_factory=list)
    roll: List[bool] = field(default_factory=list)
    last_close: List[float] = field(default_factory=list)
    move: List[np.ndarray] = field(default_factory=list)

    def add_session(self, rng: float, last_close: float, move: np.ndarray, roll: bool) -> None:
        self.ranges.append(float(rng)); self.last_close.append(float(last_close)); self.move.append(np.asarray(move, dtype=float)); self.roll.append(bool(roll))

    def add_bars(self, o, h, l, c, smin, roll: bool = False) -> None:
        self.add_session(float(np.max(h) - np.min(l)), float(c[-1]), session_move(o, c, smin), roll)

    @classmethod
    def from_sessions(cls, sess, upto: Optional[int] = None) -> "History":
        """History of the first `upto` sessions of a market_data.Sessions object (all of them by default)."""
        hs = cls()
        n = len(sess) if upto is None else upto
        for i in range(n):
            o, h, l, c, v, m = sess.day(i)
            hs.add_bars(o, h, l, c, m, bool(sess.roll[i]))
        return hs

    def context(self) -> dict:
        """Context of the next session: atr (points, NaN until 14 non-roll sessions), prev_close, sigma[12] (NaN until 14 sessions)."""
        good = [r for r, rl in zip(self.ranges, self.roll) if not rl]
        atr = float(np.mean(good[-ATR_LOOKBACK:])) if len(good) >= ATR_LOOKBACK else float("nan")
        prev_close = self.last_close[-1] if self.last_close else float("nan")
        if len(self.move) >= 14:
            with np.errstate(all="ignore"):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    sigma = np.nanmean(np.vstack(self.move[-14:]), axis=0)
        else:
            sigma = np.full(K_DEC, np.nan)
        return dict(atr=atr, prev_close=prev_close, sigma=sigma, n=len(self.ranges))
