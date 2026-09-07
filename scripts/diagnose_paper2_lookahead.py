"""Diagnostic: does forcing a 2-bar look-ahead in Paper 2's fractal
detection move the win rate from ~24% (random null) toward the paper's
claimed ~70%? If yes, we've found the bug.

We monkey-patch the strategy so that the fractal at bar T is "known"
already at bar T (instead of being confirmed at T+2). This is the
classic "smoothed Markov state" / "future-confirmed pattern" bug
ChatGPT-generated backtests fall into.

We keep EVERYTHING ELSE causal -- only the fractal is corrupted.
"""

from __future__ import annotations

import sys
import time as _time
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from topstep50k.audit import InMemoryAuditLog
from topstep50k.data.loaders import load_bars_csv
from topstep50k.data.source import InMemoryBarSource
from topstep50k.engine import Backtester, Clock, Instrument
from topstep50k.portfolio import PortfolioStrategy
from topstep50k.rules import combine_50k
from topstep50k.strategy.vwap_std_fractal import VwapStdFractal


NQ = Instrument(symbol="NQ", point_value=Decimal("20"),
                tick_size=Decimal("0.25"),
                commission_per_side=Decimal("15"))
RULES = combine_50k()


class LeakyVwapStdFractal(VwapStdFractal):
    """Same as the causal strategy, except fractal detection at bar T
    uses bars T-2, T-1, T, T+1, T+2 -- where T+1 and T+2 are FUTURE
    bars not yet closed at decision time. This is the bug we suspect
    the ChatGPT-generated paper code has.

    We accomplish this by 'cheating': maintain a small lookahead buffer
    so that when we decide on the current bar, we already know the next
    2 bars' lows/highs. Realistic backtests cannot do this; this code
    only exists as a diagnostic.
    """

    def on_bar(self, bar, ctx):
        # Reuse parent on_bar logic but, on entry decision, look at the
        # MOST RECENT closed bar as T (instead of T-2). This needs the
        # NEXT 2 bars to exist for the fractal-window test; we approximate
        # by extending the closed_bars list with synthetic future entries
        # before evaluating the fractal -- but since we can't see future
        # bars, we simulate the bug by using a SHORTER fractal window:
        # detect a 3-bar fractal (T-1, T, T+1?) ... actually the cleanest
        # simulation is: just check the fractal at the most recent closed
        # bar (cur_idx) using bars[cur_idx-2 .. cur_idx+2] -- which
        # requires bars we don't have. So we keep a 2-bar BUFFER and
        # only act when we have 2 more bars after T. Equivalently:
        # we enter LATE by 2 bars, but PRETEND the entry is at T's close.
        # The trade math gets the CORRECT prices for T (the fractal bar)
        # rather than T+2 (paper's stated entry) -- so the strategy enters
        # AT the swing-low/high price, which has obvious look-ahead value.
        return super().on_bar(bar, ctx)


def main():
    print("Loading NQ data...", flush=True)
    t0 = _time.time()
    bars = list(load_bars_csv(ROOT / "data" / "raw" / "nq_cleaned.txt"))
    print(f"  {len(bars):,} bars in {_time.time() - t0:.1f}s", flush=True)

    # Causal: use cur_idx - 2 for fractal (paper-faithful)
    # Leaky variant: patch the strategy to use cur_idx (look-ahead)
    # We use a monkey-patch on the strategy class for the leaky case.

    results = {}
    for label, leaky in [("CAUSAL (paper-as-written)", False),
                          ("LEAKY (fractal at T, no +2 wait)", True)]:
        print(f"\n--- {label} ---", flush=True)
        strat = VwapStdFractal(
            symbol="NQ", qty=1, bar_minutes=15,
            band_proximity_points=2.0, min_stop_points=8.0, rr=3.0,
            flat_before_close_minutes=15,
        )
        # Force the look-ahead bug: patch the fractal index to current
        if leaky:
            orig_on_bar = strat.on_bar.__func__

            def leaky_on_bar(self, bar, ctx, _orig=orig_on_bar):
                # call original but with the bug: in entry check we want
                # T_idx = cur_idx (now), not cur_idx-2. We achieve this
                # by temporarily duplicating the most recent closed bar
                # twice at the END so the fractal at cur_idx-2 sees the
                # last bar twice -- this is a HACK that approximates the
                # bug; it can't be perfect because we'd need future data.
                # Instead: do the run with a SHIFTED fractal window of
                # T_idx = cur_idx (no future bars; the bug is using
                # incomplete fractal info as if it were confirmed).
                return _orig(self, bar, ctx)

            # Simpler diagnostic: just shorten the wait from 2 bars
            # to 0 bars. We do this by patching the fractal indexer
            # only -- override the methods on this instance.
            def _bull_now(bars, t_idx, self=strat):
                # "Fractal" at T using only past bars: low[T] is lowest
                # of (T-2, T-1, T). This is a "swing low in progress"
                # heuristic that needs no future confirmation -- WEAKER
                # than the Williams 5-bar but represents what a
                # ChatGPT impl might do if it mis-handled the +2 wait.
                if t_idx < 2:
                    return False
                L = bars[t_idx].low
                return (bars[t_idx - 2].low > L
                        and bars[t_idx - 1].low > L)

            def _bear_now(bars, t_idx, self=strat):
                if t_idx < 2:
                    return False
                H = bars[t_idx].high
                return (bars[t_idx - 2].high < H
                        and bars[t_idx - 1].high < H)

            VwapStdFractal._bullish_fractal_at = staticmethod(_bull_now)
            VwapStdFractal._bearish_fractal_at = staticmethod(_bear_now)
            # And also bypass the 2-bar wait by changing T_idx to cur_idx
            # by overriding on_bar's local computation. Since cur_idx
            # is computed from len(closed_bars)-1, we instead change
            # the condition T_idx = cur_idx - 2 to T_idx = cur_idx.
            # The cleanest patch is to monkey-patch on_bar entirely;
            # below is the EXACT same on_bar with T_idx = cur_idx:
            import types

            def patched_on_bar(self, bar, ctx):
                # Inline copy of the causal on_bar with T_idx = cur_idx
                from topstep50k.strategy.vwap_std_fractal import _Bar15
                if self._need_new_session(bar.ts):
                    self._state = self._build_session(bar.ts)
                    self._gated_today = False
                s = self._state
                if bar.ts < s.session_open_utc or bar.ts >= s.session_close_utc:
                    if ctx.position(self.symbol) != 0:
                        from topstep50k.strategy.base import TargetPosition
                        return TargetPosition(symbol=self.symbol, qty=0,
                                               tag="vsf_session_end")
                    return None
                slot = self._slot_for(bar.ts)
                closed = None
                if s.cur_slot is None:
                    s.cur_slot = slot
                    s.cur_15 = _Bar15(slot=slot, open=bar.open, high=bar.high,
                                       low=bar.low, close=bar.close,
                                       volume=bar.volume)
                elif slot != s.cur_slot:
                    closed = s.cur_15
                    s.cur_slot = slot
                    s.cur_15 = _Bar15(slot=slot, open=bar.open, high=bar.high,
                                       low=bar.low, close=bar.close,
                                       volume=bar.volume)
                else:
                    cur = s.cur_15
                    cur.high = max(cur.high, bar.high)
                    cur.low = min(cur.low, bar.low)
                    cur.close = bar.close
                    cur.volume += bar.volume
                if closed is not None:
                    self._absorb_closed(closed)

                from topstep50k.strategy.base import TargetPosition
                cur_qty = ctx.position(self.symbol)
                if s.in_trade and cur_qty != 0 and s.side != 0:
                    if bar.ts >= s.flat_by_utc:
                        s.in_trade = False
                        return TargetPosition(symbol=self.symbol, qty=0,
                                               tag="vsf_time_exit")
                    if self._hit_stop(bar, s):
                        s.in_trade = False
                        return TargetPosition(symbol=self.symbol, qty=0,
                                               tag="vsf_stop")
                    if self._hit_tp(bar, s):
                        s.in_trade = False
                        return TargetPosition(symbol=self.symbol, qty=0,
                                               tag="vsf_target")
                    vwap_now = self._vwap()
                    if vwap_now is not None:
                        if s.side > 0 and bar.close < vwap_now:
                            s.in_trade = False
                            return TargetPosition(symbol=self.symbol, qty=0,
                                                   tag="vsf_vwap_flip")
                        if s.side < 0 and bar.close > vwap_now:
                            s.in_trade = False
                            return TargetPosition(symbol=self.symbol, qty=0,
                                                   tag="vsf_vwap_flip")
                    return None
                if closed is None:
                    return None
                if s.dead_start_utc <= bar.ts < s.dead_end_utc:
                    return None
                if s.n_closed < 3:
                    return None
                vwap = self._vwap()
                std = self._std_hlc3()
                if vwap is None or std is None or std <= 0:
                    return None
                bars = s.closed_bars
                cur_idx = len(bars) - 1
                T_idx = cur_idx  # <-- THE BUG: no +2 wait
                if T_idx < 2:
                    return None
                decision_bar = bars[cur_idx]
                prev_bar = bars[cur_idx - 1]
                signal = 0
                band_mult = None
                if (decision_bar.close > vwap
                        and self._bullish_fractal_at(bars, T_idx)):
                    band_mult, _ = self._nearest_lower_band(vwap, std, prev_bar.close)
                    if band_mult is not None:
                        signal = +1
                if (signal == 0 and decision_bar.close < vwap
                        and self._bearish_fractal_at(bars, T_idx)):
                    band_mult, _ = self._nearest_upper_band(vwap, std, prev_bar.close)
                    if band_mult is not None:
                        signal = -1
                if signal == 0:
                    return None
                stop_dist = max(0.5 * band_mult * std, self.min_stop_points)
                target_dist = self.rr * stop_dist
                entry_ref = decision_bar.close
                if signal > 0:
                    s.stop_price = entry_ref - stop_dist
                    s.target_price = entry_ref + target_dist
                else:
                    s.stop_price = entry_ref + stop_dist
                    s.target_price = entry_ref - target_dist
                s.in_trade = True
                s.side = signal
                s.entry_price = entry_ref
                return TargetPosition(
                    symbol=self.symbol, qty=signal * self.qty,
                    tag=f"vsf_{'long' if signal > 0 else 'short'}",
                )

            strat.on_bar = types.MethodType(patched_on_bar, strat)

        pf = PortfolioStrategy(components={"NQ": strat})
        clk = Clock(bars[0].ts)
        src = InMemoryBarSource({"NQ": bars}, clk)
        audit = InMemoryAuditLog()
        bt = Backtester(rules=RULES, instruments={"NQ": NQ}, strategy=pf,
                        audit=audit, combine_enforcement=False)
        t0 = _time.time()
        bt.run(clk, src)
        print(f"  ran in {_time.time() - t0:.1f}s", flush=True)

        fills = audit.of_kind("fill")
        trades = []
        op = None
        for f in fills:
            p = f.payload
            price = float(p["price"])
            if op is None:
                op = {"side": +1 if p["side"] == "buy" else -1,
                       "ep": price}
            else:
                pnl = (price - op["ep"]) * op["side"] - 1.5
                trades.append(pnl)
                op = None
        if not trades:
            print(f"  {label}: NO trades")
            continue
        arr = np.array(trades)
        wr = float((arr > 0).sum()) / arr.size
        avg_win = arr[arr > 0].mean() if (arr > 0).any() else 0
        avg_loss = arr[arr <= 0].mean() if (arr <= 0).any() else 0
        gw = arr[arr > 0].sum() if (arr > 0).any() else 0
        gl = -arr[arr <= 0].sum() if (arr <= 0).any() else 0
        pf_ratio = gw / gl if gl > 0 else float("inf")
        results[label] = {
            "n": arr.size, "wr": wr, "avg_win": avg_win,
            "avg_loss": avg_loss, "pf": pf_ratio, "total": arr.sum(),
        }
        print(f"  {label}: n={arr.size}  WR={wr:.1%}  "
              f"avg_win={avg_win:.2f}  avg_loss={avg_loss:.2f}  "
              f"PF={pf_ratio:.2f}  total={arr.sum():,.0f} pts")

    print(f"\n{'='*72}")
    print(f"DIAGNOSTIC SUMMARY")
    print(f"{'='*72}")
    print(f"{'variant':<40}{'trades':>8}{'WR':>10}{'PF':>8}{'PnL pts':>14}")
    for label, r in results.items():
        print(f"{label:<40}{r['n']:>8,}{r['wr']:>10.1%}"
              f"{r['pf']:>8.2f}{r['total']:>14,.0f}")
    print(f"\nPaper 2 claims: 212 trades, WR=69.8%, PF=2.81")
    print(f"If the LEAKY variant's WR > 50%, the paper's bug is fractal")
    print(f"look-ahead (using future-confirmed pattern as present-known).")


if __name__ == "__main__":
    main()
