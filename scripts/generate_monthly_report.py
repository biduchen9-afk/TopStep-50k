"""Generate an HTML monthly performance report for the 9-stream ensemble OOS.

Output: results/ensemble_oos_monthly.html

For each month of OOS (2025-01 → 2026-04) it shows:
  - Monthly PnL ($), # trading days, # winning/losing days
  - Max running drawdown within the month
  - Best/worst single-day PnL
  - 30-day Combine window outcomes for windows that STARTED in that month:
      pass / MLL breach / consistency cap / no target
  - Win rate (% of non-zero days that were positive)

The Combine window outcome counts give a Combine-realistic view: "if I
started a $50K Combine on any business day of this month, would I pass?"
"""

from __future__ import annotations

import sys
import time as _time
from collections import defaultdict
from datetime import date
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from topstep50k.analysis.passrate import realized_pass_rate
from topstep50k.audit import InMemoryAuditLog
from topstep50k.data.loaders import load_bars_csv
from topstep50k.data.source import InMemoryBarSource
from topstep50k.engine import Backtester, Clock, Instrument
from topstep50k.evaluation.harness import is_oos_split
from topstep50k.portfolio import PortfolioStrategy, IS_WEIGHTS
from topstep50k.regime import (
    meanrev_low_vol_gate,
    orb_expansion_gate,
    overnight_drift_post_selloff_gate,
    per_day_session_stats,
)
from topstep50k.rules import combine_50k
from topstep50k.strategy.mean_reversion import MeanReversionBollinger
from topstep50k.strategy.orb import OpeningRangeBreakout
from topstep50k.strategy.overnight_drift import OvernightDrift


ASSETS = {
    "ES": {"instrument": Instrument(symbol="ES", point_value=Decimal("50"),
                                      tick_size=Decimal("0.25"),
                                      commission_per_side=Decimal("2.50")),
           "tick_size_f": 0.25,
           "data_path": ROOT / "data" / "raw" / "es_cleaned.txt"},
    "NQ": {"instrument": Instrument(symbol="NQ", point_value=Decimal("20"),
                                      tick_size=Decimal("0.25"),
                                      commission_per_side=Decimal("2.50")),
           "tick_size_f": 0.25,
           "data_path": ROOT / "data" / "raw" / "nq_cleaned.txt"},
    "GC": {"instrument": Instrument(symbol="GC", point_value=Decimal("100"),
                                      tick_size=Decimal("0.10"),
                                      commission_per_side=Decimal("2.50")),
           "tick_size_f": 0.10,
           "data_path": ROOT / "data" / "raw" / "gc_cleaned.txt"},
}
RULES = combine_50k()
ORB_PARAMS = dict(qty=1, or_minutes=30, direction="both",
                   stop_mode="opposite_range", stop_ticks=40,
                   tp_multiple=1.0, flat_before_close_minutes=15)
MR_PARAMS  = dict(qty=1, lookback=60, sigma_mult=2.0, stop_ticks=15,
                   time_stop_minutes=45)
OD_PARAMS  = dict(qty=1, entry_offset_minutes=5, exit_offset_minutes=5)


def run_stream(bars, build_fn, asset_key):
    asset = ASSETS[asset_key]
    strat = build_fn()
    pf = PortfolioStrategy(components={asset_key: strat})
    from topstep50k.engine import Clock
    clk = Clock(bars[0].ts)
    src = InMemoryBarSource({asset_key: bars}, clk)
    bt = Backtester(rules=RULES, instruments={asset_key: asset["instrument"]},
                    strategy=pf, audit=InMemoryAuditLog(),
                    combine_enforcement=False)
    return bt.run(clk, src).daily_pnl


def to_series(daily_pnl, day_index):
    return np.array([float(daily_pnl.get(d, 0)) for d in day_index])


def month_key(d: date) -> str:
    return f"{d.year}-{d.month:02d}"


def month_label(key: str) -> str:
    y, m = key.split("-")
    name = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][int(m)]
    return f"{name} {y}"


def compute_monthly_stats(oos_arr: np.ndarray, oos_days: list[date]):
    """Per-month stats including running DD within the month."""
    by_month: dict[str, dict] = {}
    grouped: dict[str, list[tuple[date, float]]] = defaultdict(list)
    for d, v in zip(oos_days, oos_arr):
        grouped[month_key(d)].append((d, v))

    for mk, items in grouped.items():
        items.sort(key=lambda x: x[0])
        vals = np.array([v for _, v in items])
        eq = np.cumsum(vals)
        dd = (eq - np.maximum.accumulate(eq)).min() if len(eq) else 0.0
        nz = (vals != 0).sum()
        wins = (vals > 0).sum()
        losses = (vals < 0).sum()
        by_month[mk] = {
            "label": month_label(mk),
            "n_days": len(items),
            "n_traded": int(nz),
            "n_wins": int(wins),
            "n_losses": int(losses),
            "win_rate": float(wins / nz) if nz else 0.0,
            "total": float(vals.sum()),
            "max_dd": float(dd),
            "best_day": float(vals.max()) if len(vals) else 0.0,
            "worst_day": float(vals.min()) if len(vals) else 0.0,
        }
    return by_month


def compute_monthly_windows(oos_arr: np.ndarray, oos_days: list[date],
                             window_days: int = 30):
    """Count Combine 30-day window outcomes by the month they STARTED in."""
    pnl_d = {d: Decimal(str(round(float(v), 2)))
              for d, v in zip(oos_days, oos_arr)}
    rr = realized_pass_rate(pnl_d, rules=RULES,
                             starting_balance=RULES.starting_balance,
                             window_days=window_days, stride_days=1)
    by_month: dict[str, dict] = defaultdict(lambda: {
        "pass": 0, "mll_breach": 0, "consistency_fail": 0, "no_target": 0,
        "total": 0,
    })
    for w in rr.window_results:
        mk = month_key(w.start_day)
        bucket = by_month[mk]
        bucket[w.outcome] = bucket.get(w.outcome, 0) + 1
        bucket["total"] += 1
    return dict(by_month)


def render_html(by_month: dict, windows_30: dict, output_path: Path,
                 ensemble_total: float, ensemble_sharpe: float,
                 oos_start: date, oos_end: date) -> None:
    months_sorted = sorted(set(list(by_month.keys()) + list(windows_30.keys())))

    rows = []
    cumulative = 0.0
    for mk in months_sorted:
        m = by_month.get(mk, {})
        w = windows_30.get(mk, {})
        total = m.get("total", 0.0)
        cumulative += total
        pass_n = w.get("pass", 0)
        mll_n = w.get("mll_breach", 0)
        cons_n = w.get("consistency_fail", 0)
        no_t_n = w.get("no_target", 0)
        win_total = w.get("total", 0)
        pass_pct = (pass_n / win_total * 100) if win_total else 0.0

        total_class = "pos" if total > 0 else ("neg" if total < 0 else "neutral")
        dd_class = "neg" if m.get("max_dd", 0) < -500 else "neutral"
        pass_class = ("pass-good" if pass_pct >= 33
                       else "pass-ok" if pass_pct >= 15
                       else "pass-bad")

        rows.append(f"""
        <tr>
          <td class="month">{m.get('label', month_label(mk))}</td>
          <td class="num {total_class}">${total:+,.0f}</td>
          <td class="num">${cumulative:+,.0f}</td>
          <td class="num">{m.get('n_traded', 0)}/{m.get('n_days', 0)}</td>
          <td class="num">{m.get('win_rate', 0)*100:.0f}%</td>
          <td class="num pos">${m.get('best_day', 0):+,.0f}</td>
          <td class="num neg">${m.get('worst_day', 0):+,.0f}</td>
          <td class="num {dd_class}">${m.get('max_dd', 0):+,.0f}</td>
          <td class="num {pass_class}"><b>{pass_pct:.0f}%</b> ({pass_n}/{win_total})</td>
          <td class="num neg">{mll_n}</td>
          <td class="num warn">{cons_n}</td>
          <td class="num muted">{no_t_n}</td>
        </tr>""")

    rows_html = "\n".join(rows)
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>TopStep $50K Ensemble — OOS Monthly Report</title>
<style>
  body {{ background: #0e0f12; color: #e2e6ee; font: 14px/1.45 system-ui, -apple-system, Segoe UI, sans-serif;
          margin: 0; padding: 24px; }}
  h1 {{ font-size: 22px; margin: 0 0 4px; }}
  .sub {{ color: #9aa3b2; margin: 0 0 18px; font-size: 13px; }}
  .summary {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;
              margin: 20px 0; }}
  .card {{ background: #181a20; border: 1px solid #262932; border-radius: 8px;
            padding: 14px 16px; }}
  .card .label {{ color: #9aa3b2; font-size: 11px; text-transform: uppercase; letter-spacing: .04em; }}
  .card .value {{ font-size: 22px; font-weight: 600; margin-top: 6px; }}
  table {{ width: 100%; border-collapse: collapse; background: #181a20;
            border: 1px solid #262932; border-radius: 8px; overflow: hidden; }}
  th, td {{ padding: 10px 12px; text-align: left; }}
  th {{ background: #1d2029; font-weight: 600; font-size: 12px; color: #a8b0bf;
          text-transform: uppercase; letter-spacing: .04em;
          border-bottom: 1px solid #262932; }}
  td {{ border-bottom: 1px solid #20232c; font-variant-numeric: tabular-nums; }}
  td.num {{ text-align: right; }}
  td.month {{ font-weight: 600; }}
  td.pos {{ color: #4ade80; }}
  td.neg {{ color: #f87171; }}
  td.warn {{ color: #fbbf24; }}
  td.muted {{ color: #6b7280; }}
  td.neutral {{ color: #e2e6ee; }}
  td.pass-good {{ background: rgba(74, 222, 128, 0.18); color: #4ade80; }}
  td.pass-ok {{ background: rgba(251, 191, 36, 0.15); color: #fbbf24; }}
  td.pass-bad {{ background: rgba(248, 113, 113, 0.16); color: #f87171; }}
  .footer {{ margin-top: 24px; color: #6b7280; font-size: 12px; }}
  .legend {{ margin: 8px 0 18px; color: #9aa3b2; font-size: 12px; }}
  .legend span {{ display: inline-block; padding: 2px 8px; border-radius: 4px; margin-right: 6px; }}
  .legend .good {{ background: rgba(74, 222, 128, 0.18); color: #4ade80; }}
  .legend .ok {{ background: rgba(251, 191, 36, 0.15); color: #fbbf24; }}
  .legend .bad {{ background: rgba(248, 113, 113, 0.16); color: #f87171; }}
</style>
</head><body>
  <h1>TopStep $50K Ensemble — OOS Monthly Report</h1>
  <p class="sub">Out-of-Sample period: <b>{oos_start}</b> → <b>{oos_end}</b>
     · 9-stream regime-gated portfolio (ES + NQ + GC × ORB + MeanRev + OvernightDrift)
     · pass-rate-aware weights frozen from IS</p>

  <div class="summary">
    <div class="card"><div class="label">OOS Total PnL</div>
       <div class="value" style="color:#4ade80;">${ensemble_total:+,.0f}</div></div>
    <div class="card"><div class="label">OOS Sharpe</div>
       <div class="value">{ensemble_sharpe:+.2f}</div></div>
    <div class="card"><div class="label">OOS Months</div>
       <div class="value">{len(months_sorted)}</div></div>
    <div class="card"><div class="label">Profit Target</div>
       <div class="value" style="color:#9aa3b2;">$3,000 / +6%</div></div>
  </div>

  <div class="legend">
    Pass-30 cell color:
    <span class="good">≥ 33% (strong)</span>
    <span class="ok">15–33% (OK)</span>
    <span class="bad">&lt; 15% (weak)</span>
  </div>

  <table>
    <thead>
      <tr>
        <th>Month</th>
        <th class="num">PnL</th>
        <th class="num">Cum.</th>
        <th class="num">Active/Total</th>
        <th class="num">Win%</th>
        <th class="num">Best Day</th>
        <th class="num">Worst Day</th>
        <th class="num">Max DD</th>
        <th class="num">Pass-30 (start mo.)</th>
        <th class="num">MLL Breach</th>
        <th class="num">Consistency</th>
        <th class="num">No Target</th>
      </tr>
    </thead>
    <tbody>{rows_html}
    </tbody>
  </table>

  <div class="footer">
    <b>Columns explained:</b><br>
    · <b>PnL</b>: net P&L for the calendar month on the 9-stream ensemble at frozen IS weights.<br>
    · <b>Active/Total</b>: trading days with non-zero PnL / all session days in the month.<br>
    · <b>Win%</b>: percent of active days that closed positive.<br>
    · <b>Max DD</b>: deepest running drawdown vs running peak <i>within the month</i>.<br>
    · <b>Pass-30 (start mo.)</b>: of all 30-day Combine windows whose start day falls in this month, what fraction would have PASSED the $50K Combine (hit +$3K target without breaching $2K trailing MLL or the 50% consistency cap).<br>
    · <b>MLL Breach</b>: count of Combine windows starting this month that breached the $2K trailing max-loss limit.<br>
    · <b>Consistency</b>: count of windows where a single day exceeded 50% of total profit (TopStep consistency rule).<br>
    · <b>No Target</b>: count of windows where neither pass nor breach happened in 30 days (timed-out).
  </div>
</body></html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)


def main():
    print("=" * 72)
    print("MONTHLY OOS REPORT — 9-STREAM ENSEMBLE")
    print("=" * 72)

    # ── Load bars + build gates ─────────────────────────────────────────
    all_bars = {}
    gates = {}
    for ak in ASSETS:
        print(f"\nLoading {ak} bars...", flush=True)
        t0 = _time.time()
        bars = list(load_bars_csv(ASSETS[ak]["data_path"]))
        print(f"  {len(bars):,} bars in {_time.time()-t0:.1f}s")
        all_bars[ak] = bars
        stats = per_day_session_stats(bars)
        gates[ak] = {
            "orb": orb_expansion_gate(stats),
            "mr":  meanrev_low_vol_gate(stats),
            "od":  overnight_drift_post_selloff_gate(stats),
        }

    # ── Run 9 streams ──────────────────────────────────────────────────
    print("\nRunning 9 streams...", flush=True)
    streams: dict[tuple[str, str], dict] = {}
    for ak in ASSETS:
        ts = ASSETS[ak]["tick_size_f"]
        bars = all_bars[ak]
        g = gates[ak]
        t0 = _time.time()

        streams[(ak, "ORB")] = run_stream(
            bars,
            lambda gg=g, tts=ts, a=ak: OpeningRangeBreakout(
                symbol=a, tick_size=tts, daily_filter=gg["orb"], **ORB_PARAMS),
            ak)
        streams[(ak, "MeanRev")] = run_stream(
            bars,
            lambda gg=g, tts=ts, a=ak: MeanReversionBollinger(
                symbol=a, tick_size=tts, daily_filter=gg["mr"], **MR_PARAMS),
            ak)
        streams[(ak, "OD")] = run_stream(
            bars,
            lambda gg=g, a=ak: OvernightDrift(
                symbol=a, entry_filter=gg["od"], **OD_PARAMS),
            ak)
        print(f"  {ak}: 3 streams in {_time.time()-t0:.1f}s")

    # ── Build OOS ensemble PnL ─────────────────────────────────────────
    union_days = sorted(set().union(*[set(s.keys()) for s in streams.values()]))
    is_mask, oos_mask, is_days, oos_days = is_oos_split(union_days, {})

    weights = IS_WEIGHTS  # frozen from IS evaluation
    z = sum(weights.values())
    ens = np.zeros(len(union_days), dtype=float)
    for k, w in weights.items():
        if k not in streams:
            continue
        s = to_series(streams[k], union_days)
        ens += (w / z) * s
    oos_arr = ens[oos_mask]

    total = float(oos_arr.sum())
    sh = float(oos_arr.mean() / oos_arr.std(ddof=1) * np.sqrt(252)) \
         if oos_arr.std(ddof=1) > 0 else 0.0
    print(f"\nOOS total: ${total:+,.0f}  Sharpe: {sh:+.2f}  "
          f"({oos_days[0]} → {oos_days[-1]})")

    # ── Compute monthly stats ──────────────────────────────────────────
    by_month = compute_monthly_stats(oos_arr, oos_days)
    windows_30 = compute_monthly_windows(oos_arr, oos_days, window_days=30)

    # ── Render HTML ────────────────────────────────────────────────────
    out_path = ROOT / "results" / "ensemble_oos_monthly.html"
    render_html(by_month, windows_30, out_path,
                 ensemble_total=total, ensemble_sharpe=sh,
                 oos_start=oos_days[0], oos_end=oos_days[-1])
    print(f"\nHTML report written to: {out_path}")
    print(f"  Open with:  open {out_path}   (mac)   or   xdg-open ... (linux)")

    # ── Also print a plain-text monthly summary ───────────────────────
    print(f"\n{'Month':<10} {'PnL':>12} {'Cum.':>12} {'WinRate':>8} "
          f"{'MaxDD':>10} {'Pass30':>15}")
    print("─" * 72)
    cum = 0.0
    for mk in sorted(by_month.keys()):
        m = by_month[mk]; w = windows_30.get(mk, {})
        cum += m["total"]
        winrate = m["win_rate"] * 100
        passn = w.get("pass", 0); tot = w.get("total", 0)
        ppct = passn / tot * 100 if tot else 0
        print(f"{m['label']:<10} ${m['total']:>+10,.0f} ${cum:>+10,.0f} "
              f"{winrate:>6.0f}% ${m['max_dd']:>+8,.0f} "
              f"{ppct:>5.1f}% ({passn}/{tot})")


if __name__ == "__main__":
    main()
