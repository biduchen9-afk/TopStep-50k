"""Bar-by-bar (event-driven) versions of the three NQ sleeves. They consume completed one-minute bars in order, use only information available when each decision is taken
(signals are read at a bar close and filled at the next bar's open; stops fill at the stop or at the open on a gap) and reproduce, session by session, the P&L of the vectorised
back-test functions in intraday_strategies.py / academic_strategies.py / vwap_variants.py (see test_paper.py). All prices and P&L are in index POINTS for one contract; costs are
charged per round trip in points (cost_pts)."""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

CASH_OPEN = 15 * 60 + 30          # 09:30 ET in minutes after the 18:00 ET open
OR_END = CASH_OPEN + 30           # 10:00 ET
FLAT_BY = 22 * 60 - 5             # 15:55 ET
STEP = 30
K_DEC = 12


class BarSleeve:
    """Common state: one position at a time, realised points (net of costs), worst intraday mark-to-market, trade list."""
    name = "sleeve"

    def __init__(self, cost_pts: float, max_trades: int = 3):
        self.cost = cost_pts
        self.max_trades = max_trades
        self.reset(None)

    # ---- lifecycle
    def reset(self, ctx: Optional[dict]) -> None:
        self.ctx = ctx
        self.pos = 0
        self.entry = float("nan")
        self.e_idx = -1
        self.worst = 0.0               # worst (low - entry) / (entry - high) over the bars held so far in the open trade
        self.realized = 0.0
        self.minp = 0.0
        self.trades = 0
        self.trade_log: List[dict] = []
        self.done = False
        self.last_c = float("nan")
        self.last_smin = -1
        self.n_bars = 0
        self.tradable = bool(ctx is not None and ctx.get("ok", False))

    # ---- accounting helpers
    def _open(self, direction: int, price: float, i: int, smin: int) -> None:
        self.pos, self.entry, self.e_idx, self.worst = direction, price, i, 0.0
        self.trades += 1
        self._cur = dict(dir=direction, entry=price, entry_i=i, entry_smin=smin)

    def _hold(self, h: float, l: float) -> None:
        w = (l - self.entry) if self.pos > 0 else (self.entry - h)
        if w < self.worst:
            self.worst = w

    def _close(self, px: float, smin_exit: int, reason: str) -> None:
        pnl = (px - self.entry) if self.pos > 0 else (self.entry - px)
        self.minp = min(self.minp, self.realized + min(self.worst, pnl))
        self.realized += pnl - self.cost
        self._cur.update(exit=px, exit_smin=smin_exit, pnl_pts=pnl - self.cost, reason=reason)
        self.trade_log.append(self._cur)
        self.pos = 0
        self.worst = 0.0

    # ---- live views
    @property
    def day_low_pts(self) -> float:
        """Worst mark-to-market of the day so far (points, <= 0), same convention as the back-test (the open trade's cost is not charged until it closes)."""
        w = self.worst if self.pos else 0.0
        return min(self.minp, self.realized + w, 0.0) if self.pos else min(self.minp, self.realized, 0.0)

    def unrealized_pts(self, price: float) -> float:
        if not self.pos:
            return 0.0
        return (price - self.entry) if self.pos > 0 else (self.entry - price)

    def result(self):
        """(net points, worst MTM points, trades) of the finished session, as the back-test functions return."""
        return self.realized, min(self.minp, self.realized), self.trades

    def finish(self) -> None:
        """Flatten at the close of the last bar before 15:55 ET (as the back-test does) if still open."""
        if self.done:
            return
        if self.pos:
            self._close(self.last_c, self.last_smin + 1, "flat at 15:55")
        self.done = True

    def on_bar(self, i: int, smin: int, o: float, h: float, l: float, c: float, v: float) -> None:
        if self.done:
            return
        if smin >= FLAT_BY:
            self.finish()
            return
        if self.tradable:
            self._on_bar(i, smin, o, h, l, c, v)
        self.last_c, self.last_smin = c, smin
        self.n_bars += 1

    def _on_bar(self, i, smin, o, h, l, c, v):          # pragma: no cover
        raise NotImplementedError


class VwapAtrSleeve(BarSleeve):
    """VWAP+ATR momentum (intraday_strategies.vwap_atr_session): VWAP anchored at the 18:00 ET open (typical price x volume), long when close > VWAP + k*ATR, short when close <
    VWAP - k*ATR (first trade on the level, later trades on fresh crossings), stop s*ATR, exit when a close is back through the VWAP (filled at the next open), flat by 15:55.
    `start_min`: first minute-of-session at which signals are read; `direction_rule`: None (both), or a callable (ctx, state) -> +1/-1/0 evaluated once at 10:00 (gap30)."""
    name = "vwap_atr"

    def __init__(self, cost_pts, k=0.3, s=0.2, start_min=CASH_OPEN, gap30=False, max_trades=3):
        self.k, self.s, self.start_min, self.gap30 = k, s, start_min, gap30
        super().__init__(cost_pts, max_trades)

    def reset(self, ctx):
        super().reset(ctx)
        self.cpv = 0.0
        self.cv = 0.0
        self.prev_up = False
        self.prev_dn = False
        self.first = True
        self.pending_entry = 0
        self.pend_exit = False
        self.t_next = 0                 # first bar index at which signals may be read again (after an exit)
        self.allow = 0                  # +1 longs only, -1 shorts only, 0 both
        self.a_open = float("nan")      # open of the 09:30 bar
        self.started = False            # signal window open (a bar with smin >= start_min has arrived)
        self.blocked = False            # gap30: no trade today
        self.band = self.stop_d = float("nan")
        if ctx is not None and ctx.get("ok", False):
            self.band, self.stop_d = self.k * ctx["atr"], self.s * ctx["atr"]

    def _on_bar(self, i, smin, o, h, l, c, v):
        ctx = self.ctx
        if math.isnan(self.a_open) and smin >= CASH_OPEN:
            self.a_open = o
        # gap30: direction decided from the previous close, the 09:30 open and the close of the last bar before 10:00
        if self.gap30 and not self.started and smin >= self.start_min:
            sg_gap = int(np.sign(self.a_open - ctx["prev_close"])) if (not math.isnan(self.a_open) and math.isfinite(ctx["prev_close"])) else 0
            sg_30 = int(np.sign(self.last_c - self.a_open)) if (self.n_bars > 0 and not math.isnan(self.a_open) and self.last_smin >= CASH_OPEN + 5) else 0
            al = sg_gap if (sg_gap == sg_30 and sg_gap != 0) else 0
            if al == 0:
                self.blocked = True
            self.allow = al
        if not self.started and smin >= self.start_min:
            self.started = True
            self.t_next = max(self.t_next, i)
        tp = (h + l + c) / 3.0
        self.cpv += tp * v
        self.cv += v
        vw = self.cpv / max(self.cv, 1.0)
        # 1) entry filled at this bar's open from the signal read at the previous close
        if self.pending_entry and self.pos == 0:
            self._open(self.pending_entry, o, i, smin)
            self.stop = o - self.pending_entry * self.stop_d
            self.pending_entry = 0
            self.pend_exit = False
            self.first = False
        # 2) manage the open trade over this bar
        if self.pos:
            if self.pend_exit:                                   # VWAP cross at the previous close -> out at this open
                self._close(o, smin, "vwap")
                self.pend_exit = False
                self.t_next = i
            else:
                self._hold(h, l)
                hit = (l <= self.stop) if self.pos > 0 else (h >= self.stop)
                if hit:
                    px = min(self.stop, o) if self.pos > 0 else max(self.stop, o)
                    self._close(px, smin, "stop")
                    self.t_next = i + 1
                elif (self.pos > 0 and c < vw) or (self.pos < 0 and c > vw):
                    self.pend_exit = True
        # 3) signal read at this close (fill at the next open)
        up = c > vw + self.band
        dn = c < vw - self.band
        if self.allow > 0:
            dn = False
        elif self.allow < 0:
            up = False
        if (self.started and not self.blocked and self.pos == 0 and not self.pending_entry and self.trades < self.max_trades and i >= self.t_next):
            sig_up = up if self.first else (up and not self.prev_up)
            sig_dn = dn if self.first else (dn and not self.prev_dn)
            if sig_up:
                self.pending_entry = 1
            elif sig_dn:
                self.pending_entry = -1
        self.prev_up, self.prev_dn = bool(up), bool(dn)


class NoiseAreaSleeve(BarSleeve):
    """Noise-area momentum (academic_strategies.noise_area_session): every 30 minutes from 10:00 ET, flat and the last close outside [min(open, prev_close)(1-sigma), max(open, prev_close)(1+sigma)]
    -> trade the breakout at that bar's open; exit (at an open, at a 30-minute check) when the last close is back through max(upper edge, VWAP) / min(lower edge, VWAP); flat by 15:55.
    The VWAP here is anchored at the 09:30 bar."""
    name = "noise_area"

    def reset(self, ctx):
        super().reset(ctx)
        self.a0 = -1
        self.op = float("nan")
        self.hi_b = self.lo_b = float("nan")
        self.cpv = self.cv = 0.0
        self.vw_last = float("nan")
        self.k_ptr = 1

    def _on_bar(self, i, smin, o, h, l, c, v):
        ctx = self.ctx
        if self.a0 < 0 and smin >= CASH_OPEN:
            self.a0, self.op = i, o
            pc = ctx["prev_close"]
            self.hi_b, self.lo_b = max(self.op, pc), min(self.op, pc)
        # decisions due at this bar (the first bar at or after each 30-minute mark)
        while self.k_ptr <= K_DEC and smin >= CASH_OPEN + STEP * self.k_ptr and self.a0 >= 0 and i > self.a0:
            k = self.k_ptr
            self.k_ptr += 1
            sg = ctx["sigma"][k - 1]
            if not math.isfinite(sg):
                continue
            px = self.last_c
            vwap = self.vw_last if self.cv > 0 else self.op
            ub, lb = self.hi_b * (1 + sg), self.lo_b * (1 - sg)
            if self.pos == 0 and self.trades < self.max_trades:
                if px > ub:
                    self._open(1, o, i, smin)
                elif px < lb:
                    self._open(-1, o, i, smin)
            elif self.pos > 0 and px < max(ub, vwap):
                self._close(o, smin, "noise-area exit")
            elif self.pos < 0 and px > min(lb, vwap):
                self._close(o, smin, "noise-area exit")
        if self.pos:
            self._hold(h, l)
        if self.a0 >= 0:
            self.cpv += (h + l + c) / 3.0 * v
            self.cv += v
            self.vw_last = self.cpv / max(self.cv, 1.0)


def make_sleeves(cost_pts: float) -> dict:
    """The three sleeves of the final strategy, keyed as in final_strategy.py (nq_noise, nq_vwap, nq_gap30)."""
    return {"nq_noise": NoiseAreaSleeve(cost_pts), "nq_vwap": VwapAtrSleeve(cost_pts, 0.3, 0.2, CASH_OPEN, False),
            "nq_gap30": VwapAtrSleeve(cost_pts, 0.3, 0.2, OR_END, True)}
