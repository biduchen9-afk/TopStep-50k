"""Strategy #11 (re-run): Sweep → MSS/Impulse with Pine-style same-bar-close fills.

Tests the user's hypothesis: does executing fills at the SIGNAL bar's close
(matching TradingView Pine Script `process_orders_on_close=true`) change
the verdict from the conservative next-bar-open fill model?

Implementation: standalone simulator (no engine). For each signal bar:
  ENTRY  : entry_price = 5-min signal bar's CLOSE  (Pine-style)
  STOP   : intra-bar touch detected on subsequent 1-min bars (first bar
           that breaches → fill at the stop price)
  TARGET : intra-bar touch on subsequent 1-min bars → fill at target price
  EXIT   : force close at 23:59 Bangkok → fill at that 1-min bar's close

Commission = $2.50/side × 2 = $5/round-trip per contract.
"""

from __future__ import annotations

import sys
import time as _time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Deque
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from topstep50k.analysis.passrate import realized_pass_rate
from topstep50k.data.loaders import load_bars_csv
from topstep50k.engine.types import Bar
from topstep50k.evaluation.harness import is_oos_split
from topstep50k.rules import combine_50k

BANGKOK = ZoneInfo("Asia/Bangkok")
GC_TICK = 0.10
GC_POINT_VALUE = 100.0
COMMISSION_PER_SIDE = 2.50

LONDON_START = time(12, 0)
LONDON_END   = time(14, 0)
NY_START     = time(18, 0)
NY_END       = time(19, 30)
FORCE_CLOSE  = time(23, 59)

RR              = 2.0
ATR_LEN         = 14
ATR_MULT        = 0.65
CLOSE_IN_EXTREME = 0.65
MSS_LOOKBACK    = 5
MAX_WAIT_BARS   = 24

RULES = combine_50k()


@dataclass
class Bar5m:
    ts_end: datetime
    open: float
    high: float
    low: float
    close: float


def aggregate_5m(bars: list[Bar]) -> tuple[list[Bar5m], list[list[Bar]]]:
    """Returns (5m bars, list of 1m bars belonging to each 5m slot)."""
    out_5m: list[Bar5m] = []
    out_1m_groups: list[list[Bar]] = []
    cur: list[Bar] = []
    cur_key = None

    def slot_key(ts: datetime) -> tuple:
        m = ts.minute
        if m == 0:
            return (ts.year, ts.month, ts.day, ts.hour, 0)
        end_min = ((m - 1) // 5 + 1) * 5
        if end_min == 60:
            nxt = ts.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            return (nxt.year, nxt.month, nxt.day, nxt.hour, 0)
        return (ts.year, ts.month, ts.day, ts.hour, end_min)

    for b in bars:
        k = slot_key(b.ts)
        if cur_key is None:
            cur_key = k
            cur.append(b)
            continue
        if k != cur_key:
            # close out current
            out_5m.append(Bar5m(
                ts_end=cur[-1].ts,
                open=cur[0].open,
                high=max(x.high for x in cur),
                low=min(x.low for x in cur),
                close=cur[-1].close,
            ))
            out_1m_groups.append(cur)
            cur = [b]
            cur_key = k
        else:
            cur.append(b)
    if cur:
        out_5m.append(Bar5m(
            ts_end=cur[-1].ts,
            open=cur[0].open,
            high=max(x.high for x in cur),
            low=min(x.low for x in cur),
            close=cur[-1].close,
        ))
        out_1m_groups.append(cur)
    return out_5m, out_1m_groups


def compute_atr(bars5m: list[Bar5m], length: int) -> list[float | None]:
    out: list[float | None] = [None] * len(bars5m)
    trs: Deque[float] = deque(maxlen=length)
    prev_close: float | None = None
    for i, b in enumerate(bars5m):
        if prev_close is None:
            tr = b.high - b.low
        else:
            tr = max(b.high - b.low,
                     abs(b.high - prev_close),
                     abs(b.low - prev_close))
        trs.append(tr)
        out[i] = sum(trs) / len(trs)
        prev_close = b.close
    return out


@dataclass
class Trade:
    entry_ts: datetime
    exit_ts: datetime
    side: int  # +1 long, -1 short
    entry: float
    exit: float
    reason: str  # 'tp', 'sl', 'force_close'

    @property
    def pnl(self) -> float:
        gross = (self.exit - self.entry) * self.side * GC_POINT_VALUE
        return gross - 2 * COMMISSION_PER_SIDE  # round-trip commission


def find_exit(bars1m: list[Bar], start_idx: int, side: int,
              stop: float, target: float,
              force_close_dt: datetime) -> tuple[datetime, float, str]:
    """Scan forward 1m bars to find the first stop/target/force-close hit."""
    for i in range(start_idx, len(bars1m)):
        b = bars1m[i]
        if b.ts >= force_close_dt:
            return b.ts, b.close, "force_close"
        if side > 0:
            if b.low <= stop:
                return b.ts, stop, "sl"
            if b.high >= target:
                return b.ts, target, "tp"
        else:
            if b.high >= stop:
                return b.ts, stop, "sl"
            if b.low <= target:
                return b.ts, target, "tp"
    # ran out of data
    return bars1m[-1].ts, bars1m[-1].close, "data_end"


def simulate(bars1m: list[Bar], qty: int = 1) -> tuple[list[Trade], dict[date, float]]:
    """Pine-style same-bar-close-fill simulator. Returns trades and daily PnL."""
    bars5m, groups = aggregate_5m(bars1m)
    atrs = compute_atr(bars5m, ATR_LEN)

    # Build a flat 1m index for forward-scanning at exit time
    bars1m_sorted = bars1m  # already sorted by load_bars_csv

    # Map 5m index → first 1m bar index in the NEXT 5m group (where exit-scan starts)
    # We need the absolute 1m position for each 5m. Build a cumulative index map.
    next_1m_index: list[int] = []
    cum = 0
    for g in groups:
        cum += len(g)
        next_1m_index.append(cum)  # index of FIRST 1m bar AFTER this 5m group

    # Track per-Bangkok-day trading state
    trades: list[Trade] = []
    daily_pnl: dict[date, float] = defaultdict(float)

    cur_bkk_day: date | None = None
    london_high = london_low = ny_high = ny_low = None
    london_swept_high = london_swept_low = ny_swept_high = ny_swept_low = False
    london_sweep_high_px = london_sweep_low_px = ny_sweep_high_px = ny_sweep_low_px = None
    london_sweep_5m_idx = ny_sweep_5m_idx = None
    london_traded = ny_traded = False
    in_position = False
    position_exit_idx_1m = -1  # 1m absolute index after which we can re-enter
    prev_bear_sig = False
    prev_bull_sig = False

    # We need to update sweep state at 1m granularity (sweep can happen anywhere
    # post-window), and update london/ny range during the window also at 1m.
    # But signals fire at 5m close. So we walk per-5m and consult the 1m group
    # for sweep detection.

    for i5, b5 in enumerate(bars5m):
        group = groups[i5]  # 1m bars in this 5m slot
        first_1m_abs = next_1m_index[i5 - 1] if i5 > 0 else 0
        # last 1m bar of this 5m is at index (next_1m_index[i5] - 1)
        last_1m_abs = next_1m_index[i5] - 1

        bkk = group[-1].ts.astimezone(BANGKOK)
        bkk_day = bkk.date()

        if cur_bkk_day != bkk_day:
            # Reset daily state
            cur_bkk_day = bkk_day
            london_high = london_low = ny_high = ny_low = None
            london_swept_high = london_swept_low = ny_swept_high = ny_swept_low = False
            london_sweep_high_px = london_sweep_low_px = None
            ny_sweep_high_px = ny_sweep_low_px = None
            london_sweep_5m_idx = ny_sweep_5m_idx = None
            london_traded = ny_traded = False
            prev_bear_sig = False
            prev_bull_sig = False

        # Walk 1m bars inside this 5m to track window highs/lows and sweeps
        for b1 in group:
            bkk_t = b1.ts.astimezone(BANGKOK).time()
            in_london = LONDON_START <= bkk_t < LONDON_END
            in_ny = NY_START <= bkk_t < NY_END

            if in_london:
                london_high = b1.high if london_high is None else max(london_high, b1.high)
                london_low  = b1.low  if london_low  is None else min(london_low, b1.low)
            if in_ny:
                ny_high = b1.high if ny_high is None else max(ny_high, b1.high)
                ny_low  = b1.low  if ny_low  is None else min(ny_low, b1.low)

            past_london = bkk_t >= LONDON_END
            past_ny     = bkk_t >= NY_END

            if past_london and london_high is not None and not london_traded:
                if not london_swept_high and b1.high > london_high:
                    london_swept_high = True
                    london_sweep_high_px = b1.high
                    london_sweep_5m_idx = i5
                if not london_swept_low and b1.low < london_low:
                    london_swept_low = True
                    london_sweep_low_px = b1.low
                    london_sweep_5m_idx = i5

            if past_ny and ny_high is not None and not ny_traded:
                if not ny_swept_high and b1.high > ny_high:
                    ny_swept_high = True
                    ny_sweep_high_px = b1.high
                    ny_sweep_5m_idx = i5
                if not ny_swept_low and b1.low < ny_low:
                    ny_swept_low = True
                    ny_sweep_low_px = b1.low
                    ny_sweep_5m_idx = i5

        # Now evaluate signals at the close of this 5m bar
        if in_position:
            # we're already inside a trade — skip
            continue
        if last_1m_abs <= position_exit_idx_1m:
            # haven't moved past the exit bar yet (shouldn't happen since we set
            # in_position=False on close)
            continue

        atr = atrs[i5]
        if atr is None or i5 < MSS_LOOKBACK + 1:
            prev_bear_sig = False
            prev_bull_sig = False
            continue

        body = abs(b5.close - b5.open)
        rng = max(b5.high - b5.low, GC_TICK)
        close_pct = (b5.close - b5.low) / rng

        bear_impulse = (b5.close < b5.open and body >= atr * ATR_MULT
                         and close_pct <= 1.0 - CLOSE_IN_EXTREME)
        bull_impulse = (b5.close > b5.open and body >= atr * ATR_MULT
                         and close_pct >= CLOSE_IN_EXTREME)

        prior = bars5m[i5 - MSS_LOOKBACK : i5]
        prior_high = max(p.high for p in prior)
        prior_low  = min(p.low  for p in prior)

        bear_sig = bear_impulse and (b5.close < prior_low)
        bull_sig = bull_impulse and (b5.close > prior_high)

        new_bear = bear_sig and not prev_bear_sig
        new_bull = bull_sig and not prev_bull_sig
        prev_bear_sig = bear_sig
        prev_bull_sig = bull_sig

        if not (new_bear or new_bull):
            continue

        def bars_since(ix):
            return None if ix is None else i5 - ix

        valid_ny_short = (not ny_traded and ny_swept_high
                          and bars_since(ny_sweep_5m_idx) is not None
                          and bars_since(ny_sweep_5m_idx) <= MAX_WAIT_BARS
                          and new_bear)
        valid_london_short = (not london_traded and london_swept_high
                              and bars_since(london_sweep_5m_idx) is not None
                              and bars_since(london_sweep_5m_idx) <= MAX_WAIT_BARS
                              and new_bear)
        valid_ny_long = (not ny_traded and ny_swept_low
                         and bars_since(ny_sweep_5m_idx) is not None
                         and bars_since(ny_sweep_5m_idx) <= MAX_WAIT_BARS
                         and new_bull)
        valid_london_long = (not london_traded and london_swept_low
                             and bars_since(london_sweep_5m_idx) is not None
                             and bars_since(london_sweep_5m_idx) <= MAX_WAIT_BARS
                             and new_bull)

        short_entry = valid_ny_short or valid_london_short
        long_entry  = valid_ny_long  or valid_london_long
        if not (short_entry or long_entry):
            continue

        entry_px = b5.close
        entry_ts = b5.ts_end

        # Force close at 23:59 Bangkok today (Bangkok day)
        # Note: a Bangkok day starts at 00:00 Bangkok = 17:00 UTC prev day
        bkk_today = entry_ts.astimezone(BANGKOK).date()
        force_close_local = datetime.combine(bkk_today, FORCE_CLOSE, tzinfo=BANGKOK)
        force_close_utc = force_close_local.astimezone(entry_ts.tzinfo)

        if long_entry:
            sweep_low = (ny_sweep_low_px if valid_ny_long else london_sweep_low_px)
            stop = min(sweep_low, prior_low)
            risk = entry_px - stop
            if risk <= GC_TICK:
                continue
            target = entry_px + risk * RR
            stop   = round(stop   / GC_TICK) * GC_TICK
            target = round(target / GC_TICK) * GC_TICK
            side = +1
            if valid_ny_long: ny_traded = True
            else: london_traded = True
        else:
            sweep_high = (ny_sweep_high_px if valid_ny_short else london_sweep_high_px)
            stop = max(sweep_high, prior_high)
            risk = stop - entry_px
            if risk <= GC_TICK:
                continue
            target = entry_px - risk * RR
            stop   = round(stop   / GC_TICK) * GC_TICK
            target = round(target / GC_TICK) * GC_TICK
            side = -1
            if valid_ny_short: ny_traded = True
            else: london_traded = True

        in_position = True
        # Scan forward 1m bars starting AFTER this 5m's last 1m bar
        exit_start_1m = next_1m_index[i5]
        exit_ts, exit_px, reason = find_exit(
            bars1m_sorted, exit_start_1m, side, stop, target, force_close_utc
        )
        tr = Trade(entry_ts=entry_ts, exit_ts=exit_ts, side=side,
                   entry=entry_px, exit=exit_px, reason=reason)
        trades.append(tr)
        pnl_unit = tr.pnl
        daily_pnl[exit_ts.astimezone(BANGKOK).date()] += pnl_unit * qty
        in_position = False
        # Mark the 5m idx beyond which the exit 1m has been consumed
        # (next signal evaluation is on the following 5m bars naturally)

    return trades, dict(daily_pnl)


def main():
    print("=" * 72)
    print("STRATEGY #11 (RE-RUN): SWEEP → MSS — PINE-STYLE SAME-BAR FILLS")
    print("Entry = signal bar's CLOSE; stops/targets touch on subsequent 1m bars")
    print("=" * 72)

    print("\nLoading GC bars...", flush=True)
    t0 = _time.time()
    bars = list(load_bars_csv(ROOT / "data" / "raw" / "gc_cleaned.txt"))
    print(f"  {len(bars):,} bars in {_time.time()-t0:.1f}s", flush=True)

    print("\nSimulating with Pine-style fills (1 contract)...", flush=True)
    t0 = _time.time()
    trades, daily_pnl = simulate(bars, qty=1)
    print(f"  {len(trades)} trades, {len(daily_pnl)} active days in "
          f"{_time.time()-t0:.1f}s", flush=True)

    union_days = sorted(daily_pnl.keys()) or [bars[0].ts.date(), bars[-1].ts.date()]
    # Need a regular daily index from bars to align with the engine framing
    all_days = sorted({b.ts.astimezone(BANGKOK).date() for b in bars})
    is_mask, oos_mask, is_days, oos_days = is_oos_split(all_days, {})

    def to_arr(days):
        return np.array([daily_pnl.get(d, 0.0) for d in days])

    is_arr  = to_arr(is_days)
    oos_arr = to_arr(oos_days)

    def stats(label, arr, days):
        if arr.size == 0:
            print(f"  {label}: empty"); return
        sh = (arr.mean() / arr.std(ddof=1) * np.sqrt(252)) if arr.std(ddof=1) > 0 else 0
        pf_v = (arr[arr>0].sum() / -arr[arr<0].sum()) if (arr<0).any() else float("inf")
        eq = np.cumsum(arr); mdd = float((eq - np.maximum.accumulate(eq)).min())
        nz = int((arr != 0).sum())
        print(f"  {label:<35}: n={arr.size} nz={nz} total=${arr.sum():+,.0f} "
              f"EV=${arr.mean():+.1f}/day Sharpe={sh:+.2f} PF={pf_v:.2f} "
              f"MaxDD=${mdd:+,.0f}")
        pnl_d = {d: Decimal(str(round(float(v),2))) for d,v in zip(days, arr)}
        rr30 = realized_pass_rate(pnl_d, rules=RULES,
                                   starting_balance=RULES.starting_balance,
                                   window_days=30, stride_days=1)
        rr45 = realized_pass_rate(pnl_d, rules=RULES,
                                   starting_balance=RULES.starting_balance,
                                   window_days=45, stride_days=1)
        print(f"  {'':>35}  pass30={rr30.pass_rate:.1%} ({rr30.n_passed}/{rr30.n_windows})"
              f"  pass45={rr45.pass_rate:.1%} ({rr45.n_passed}/{rr45.n_windows})")
        oc = rr30.outcomes
        print(f"  {'':>35}  30d outcomes: pass={oc.get('pass',0)} "
              f"mll={oc.get('mll_breach',0)} cons={oc.get('consistency_fail',0)} "
              f"no_target={oc.get('no_target',0)}")

    print(f"\nIS : {is_days[0]} → {is_days[-1]} ({len(is_days)}d)")
    print(f"OOS: {oos_days[0]} → {oos_days[-1]} ({len(oos_days)}d)\n")

    print("─" * 72)
    print("IS (1 contract, same-bar-close fills)")
    print("─" * 72)
    stats("IS Pine-fill", is_arr, is_days)

    print("\n" + "─" * 72)
    print("OOS (1 contract, same-bar-close fills)")
    print("─" * 72)
    stats("OOS Pine-fill", oos_arr, oos_days)

    # Trade-level breakdown
    print("\n" + "─" * 72)
    print("TRADE-LEVEL BREAKDOWN (full history)")
    print("─" * 72)
    by_reason: dict[str, int] = defaultdict(int)
    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    for t in trades:
        by_reason[t.reason] += 1
    print(f"  Total trades: {len(trades)}")
    print(f"  Win rate: {len(wins)/len(trades)*100:.1f}% ({len(wins)}/{len(trades)})")
    if wins: print(f"  Avg win: ${np.mean(wins):+,.0f}  Max win: ${max(wins):+,.0f}")
    if losses: print(f"  Avg loss: ${np.mean(losses):+,.0f}  Max loss: ${min(losses):+,.0f}")
    if losses and wins:
        print(f"  Profit factor: {sum(wins) / -sum(losses):.3f}")
    print(f"  Exit reasons: {dict(by_reason)}")


if __name__ == "__main__":
    main()
