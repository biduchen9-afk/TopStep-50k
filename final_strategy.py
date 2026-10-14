"""The strategy and farm configuration that came out of the study, in one place (see FINDINGS.md for the evidence and the caveats).

SIGNALS - three NQ sleeves, all long/short, all flat by 15:55 ET, signals read at a bar close and filled at the next open
(micro NQ, $1.22 all-in per micro round turn + 1 tick slippage per round turn):
  nq_noise  noise-area momentum (Zarattini-Aziz-Barbon style): every 30 min from 10:00 ET, flat and price outside the "noise area"
            [min(open, prev_close)*(1-sigma_t), max(open, prev_close)*(1+sigma_t)] (sigma_t = 14-day mean |move from open| at that time of day)
            -> trade the breakout; exit when price falls back through max(upper edge, VWAP) (long) / min(lower edge, VWAP) (short).  <= 3 trades/day.
  nq_vwap   VWAP+ATR momentum: close > session VWAP + 0.3*ATR14 (long) / < VWAP - 0.3*ATR14 (short) after 09:30 ET; stop 0.2*ATR14; exit when price
            closes back through VWAP.  <= 3 trades/day.  (VWAP anchored at the 18:00 ET session open.)
  nq_gap30  the same VWAP+ATR entry (k=0.3, s=0.2) but only in the direction that BOTH the overnight gap (09:30 open vs previous close) and the first
            30-minute move (09:30-10:00) agree on; signals start at 10:00 ET; otherwise no trade that day.
PORTFOLIO - 1-micro baskets with long-only mean-variance weights fitted on 2010-2022 only (shrunk 50% to equal weights); static (no switching).
SIZING - Combine: micros = round(R / ATR$ per micro) with R = 9,600 ($ per ATR unit); funded: micros = a * (balance - MLL floor) / ATR$ with a1 = 0.4 before the first
         payout and a2 = 1.6 afterwards (0.4 pays more often at similar dev net; see the monthly-payout study), capped by the XFA scaling ladder.
FARM   - up to 5 funded accounts; new Combines only when expected funded accounts < 5; "hyst" refill (once full, open nothing until 3 slots have emptied);
         first payout at balance >= $3,000 with 5 winning days of >= $150 (Standard path), then every eligible payout at 50% of balance (cap $5,000).
FEES   - no_activation_dll path: $199 per 21 sessions per Combine, no activation fee, -$3,000 daily loss limit.

Run `python final_strategy.py` (needs the Databento NQ/ES/GC 1-minute data in /home/user/data_raw; see README) to rebuild the stream and re-run the farm.
"""
from __future__ import annotations

import numpy as np

import intraday_strategies as st
from contract_eval import CombineSizing
from farm_real import FarmSpec, run_farm
from portfolio import basket, dev_weights, make_sleeve
from real_engine import FundedPlan, Stream, costs_from_config, rules_from_config
from pricing_config import load_config

PATH = "no_activation_dll"
FINAL = dict(R=9600.0, a1=0.4, a2=1.6, t1=3000.0, refill="hyst", pause_below=0.0, max_funded=5, max_evals=10, p_est=1.0)


def build_stream() -> Stream:
    """The three-sleeve NQ basket with dev-only weights (data through 2022-12-31)."""
    cfg = load_config()
    costs = costs_from_config(cfg)
    c = st.CASH_OPEN
    sleeves = [make_sleeve("nq", "noise_area", {}, costs=costs, name="nq_noise"),
               make_sleeve("nq", "vwap_atr", dict(k=0.3, s=0.2, start_min=c), costs=costs, name="nq_vwap"),
               make_sleeve("nq", "vv_filt", dict(filt="gap30", k=0.3, s=0.2), costs=costs, name="nq_gap30")]
    weights = dev_weights(sleeves, shrink=0.5)
    return basket(sleeves, weights=weights, name="final")


def farm_from(stream: Stream, start: str, n_end: int | None = None, **overrides) -> dict:
    p = dict(FINAL, **overrides)
    cfg = load_config()
    rules = rules_from_config(cfg, PATH)
    plan = FundedPlan(f"standard t1={p['t1']} a1={p['a1']} a2={p['a2']}", "standard", p["t1"], 0.5, 1.0, 0.0, 0.5, 1.0, 99, p["a1"], p["a2"])
    spec = FarmSpec(max_evals=p["max_evals"], max_funded=p["max_funded"], p_est=p["p_est"], refill=p["refill"], pause_below=p["pause_below"])
    i0 = int(np.searchsorted(stream.dates, np.datetime64(start)))
    return run_farm([stream], CombineSizing("fixed", p["R"]), None, plan, cfg, rules, PATH, i0, spec, n_end=n_end)


if __name__ == "__main__":
    s = build_stream()
    first = str(s.dates[int(np.flatnonzero(s.ok)[0])])
    print(f"{'start':<12}{'net':>10}{'payouts':>10}{'fees':>9}{'funded':>8}")
    for d in (first, "2018-01-02", "2021-01-04", "2023-01-02", "2024-01-02", "2025-01-02", "2025-07-01"):
        r = farm_from(s, d)
        print(f"{d:<12}{r['net']:>10,.0f}{r['payouts']:>10,.0f}{r['fees']:>9,.0f}{r['n_funded']:>8d}")
