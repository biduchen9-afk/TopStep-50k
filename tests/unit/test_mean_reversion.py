"""Bollinger mean-reversion strategy unit tests.

Properties asserted:
  * No entry without sufficient lookback history.
  * No entry outside trade-window (before trade_start_local).
  * Short entry fires AFTER a spike above the upper band returns to it.
  * Long entry fires AFTER a dip below the lower band returns to it.
  * One trade per day (no re-entry).
  * EOD forces flat regardless of S/T.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from decimal import Decimal

from topstep50k.audit import InMemoryAuditLog
from topstep50k.data.source import InMemoryBarSource
from topstep50k.engine import Backtester, Bar, Clock, Instrument
from topstep50k.portfolio import PortfolioStrategy
from topstep50k.rules import combine_50k
from topstep50k.strategy.mean_reversion import MeanReversionBollinger


ES = Instrument(symbol="ES", point_value=Decimal("50"),
                tick_size=Decimal("0.25"),
                commission_per_side=Decimal("0"))


def _utc(y, m, d, h, mi=0):
    return datetime(y, m, d, h, mi, tzinfo=timezone.utc)


def _bar(ts, price, hi=None, lo=None):
    return Bar(ts=ts, open=price,
               high=hi if hi is not None else price + 0.25,
               low=lo if lo is not None else price - 0.25,
               close=price, volume=100)


def _run(strategy, bars, start_ts):
    clk = Clock(start_ts - timedelta(seconds=1))
    data = InMemoryBarSource({"ES": bars}, clk)
    pf = PortfolioStrategy(components={"ES": strategy})
    bt = Backtester(rules=combine_50k(), instruments={"ES": ES},
                    strategy=pf, audit=InMemoryAuditLog(),
                    combine_enforcement=False)
    return bt.run(clk, data)


def _quiet_session_with_spike(spike_up: bool, spike_at_minute: int = 90):
    """Build a single RTH session of 1-min bars where most of the day
    is flat at 4500 (zero variance) but one bar spikes high or low,
    then returns into the band on the next bar.

    With sigma=0 for a constant series, ANY deviation is > sigma * 2,
    so we add a tiny background jitter so sigma is small but positive.

    June 3 2024 (EDT): 09:30 ET = 13:30 UTC, close 16:00 ET = 20:00 UTC.
    390 bars total.
    """
    bars = []
    open_utc = _utc(2024, 6, 3, 13, 30)
    base = 4500.0
    for i in range(390):
        ts = open_utc + timedelta(minutes=i)
        # Light jitter to make sigma well-defined: +/- 0.25 alternating
        jitter = 0.25 if i % 2 == 0 else -0.25
        p = base + jitter
        if i == spike_at_minute:
            p = base + (5.0 if spike_up else -5.0)  # big spike
        elif i == spike_at_minute + 1:
            p = base  # snap back inside band
        bars.append(_bar(ts, p))
    return bars


def test_no_entry_before_trade_window():
    bars = _quiet_session_with_spike(spike_up=True, spike_at_minute=10)
    # Spike at minute 10 -> 09:40 ET, before trade_start_local=10:00.
    strat = MeanReversionBollinger(symbol="ES", lookback=20,
                                    trade_start_local=time(10, 0))
    result = _run(strat, bars, start_ts=_utc(2024, 6, 3, 13, 30))
    fills = [f for f in result.audit.of_kind("fill")
             if "forced" not in f.payload]
    assert fills == [], "should not enter before trade window"


def test_no_entry_without_lookback_warmup():
    bars = _quiet_session_with_spike(spike_up=True, spike_at_minute=5)
    strat = MeanReversionBollinger(symbol="ES", lookback=60,
                                    trade_start_local=time(9, 30))
    result = _run(strat, bars, start_ts=_utc(2024, 6, 3, 13, 30))
    # spike at minute 5; lookback is 60 -> not warmed up yet
    fills = [f for f in result.audit.of_kind("fill")
             if "forced" not in f.payload]
    assert fills == []


def test_upper_spike_then_return_fires_short():
    bars = _quiet_session_with_spike(spike_up=True, spike_at_minute=90)
    strat = MeanReversionBollinger(symbol="ES", lookback=30,
                                    sigma_mult=1.5,
                                    trade_start_local=time(10, 0))
    result = _run(strat, bars, start_ts=_utc(2024, 6, 3, 13, 30))
    fills = [f for f in result.audit.of_kind("fill")
             if "forced" not in f.payload]
    sides = [f.payload["side"] for f in fills]
    assert sides, "should have at least one fill"
    assert sides[0] == "sell", f"first fill should be SELL (mr_short), got {sides}"


def test_lower_spike_then_return_fires_long():
    bars = _quiet_session_with_spike(spike_up=False, spike_at_minute=90)
    strat = MeanReversionBollinger(symbol="ES", lookback=30,
                                    sigma_mult=1.5,
                                    trade_start_local=time(10, 0))
    result = _run(strat, bars, start_ts=_utc(2024, 6, 3, 13, 30))
    fills = [f for f in result.audit.of_kind("fill")
             if "forced" not in f.payload]
    sides = [f.payload["side"] for f in fills]
    assert sides and sides[0] == "buy"


def test_one_trade_per_day():
    # Two spikes in same day -- only the first should trade.
    bars = _quiet_session_with_spike(spike_up=True, spike_at_minute=90)
    # Inject a second spike at minute 200
    open_utc = _utc(2024, 6, 3, 13, 30)
    bars[200] = _bar(open_utc + timedelta(minutes=200), 4510.0)
    bars[201] = _bar(open_utc + timedelta(minutes=201), 4500.0)
    strat = MeanReversionBollinger(symbol="ES", lookback=30, sigma_mult=1.5,
                                    trade_start_local=time(10, 0))
    result = _run(strat, bars, start_ts=_utc(2024, 6, 3, 13, 30))
    fills = [f for f in result.audit.of_kind("fill")
             if "forced" not in f.payload]
    # one entry + at most one exit = 2 fills max
    # the second spike must not produce a fresh entry
    entries = [f for f in fills if "mr_short" in f.payload.get("tag", "")
               or "mr_long" in f.payload.get("tag", "")]
    assert len(entries) <= 1


def test_eod_forces_flat():
    bars = _quiet_session_with_spike(spike_up=True, spike_at_minute=90)
    strat = MeanReversionBollinger(symbol="ES", lookback=30, sigma_mult=1.5,
                                    trade_start_local=time(10, 0),
                                    # huge time stop so it relies on EOD
                                    time_stop_minutes=600,
                                    stop_ticks=10000)
    result = _run(strat, bars, start_ts=_utc(2024, 6, 3, 13, 30))
    fills = result.audit.of_kind("fill")
    net = 0
    for f in fills:
        p = f.payload
        if p["side"] == "buy":
            net += p["qty"]
        else:
            net -= p["qty"]
    assert net == 0, "must end flat after EOD"
