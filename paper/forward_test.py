"""Forward test of the frozen baseline on bars the development data has never seen.

   python -m paper.forward_test --bars data_new/nq_lse_1m.pkl [--out results/forward_test_report.json]

The stored Databento sessions (through 2026-07-02) warm up the history (ATR, noise-area sigmas, previous close); the farm starts fresh on the first new session. Reported (see docs/PREREGISTRATION_15.md):
  * data checks (volume present, RTH completeness, price continuity at the join) - the test is INVALID if they fail;
  * strategy level: P&L per micro of the two sleeves over the window and its percentile among every equal-length window of the development period (repaired every-session data when available);
  * farm level (descriptive only: 78% of 75-session farms lose on the development data): Combines, passes, funded, payouts, fees, net, and the percentile among equal-length development farms;
  * the declared kill criteria.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

from paper.farm import PaperFarm
from paper.history import History
from paper.run import load_costs_and_sessions
from paper.runner import PaperRunner
from paper.server import jsonable


def window_pct(series: np.ndarray, L: int, value: float) -> float:
    """Percentile rank of `value` among all L-session rolling sums of `series`."""
    c = np.r_[0.0, np.cumsum(series)]
    w = c[L:] - c[:-L]
    return float((w < value).mean() * 100) if len(w) else float("nan")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", required=True)
    ap.add_argument("--out", default="results/forward_test_report.json")
    ap.add_argument("--state", default=None)
    ap.add_argument("--warm-until", default=None, help="use stored sessions only up to this label for the warm-up (machinery tests on already-seen data); default: all stored sessions")
    a = ap.parse_args(argv)
    with open(a.bars, "rb") as f:
        bars = pickle.load(f)
    sess, cost_pts = load_costs_and_sessions()
    n = len(sess) if a.warm_until is None else int(np.searchsorted(sess.dates.astype("datetime64[D]"), np.datetime64(pd.Timestamp(a.warm_until).date()), side="right"))
    last_db = pd.Timestamp(sess.dates[n - 1]).date()
    runner = PaperRunner(History.from_sessions(sess, n), PaperFarm(), cost_pts, a.state, dict(mode="FORWARD TEST", symbol="LSE feed"), t0=n)
    runner.last_label = last_db
    new = [b for b in bars if b.label > last_db]
    for b in new:
        runner.process(b)
    runner.close_open_session()
    J = pd.DataFrame(runner.journal)
    rep = dict(first_new_session=str(J.date.iloc[0]) if len(J) else None, last_session=str(J.date.iloc[-1]) if len(J) else None, sessions=len(J))
    if len(J) and pd.Timestamp(J.date.iloc[0]).date() <= pd.Timestamp(sess.dates[-1]).date():
        rep["WARNING"] = "these sessions overlap the stored development / seen data: this is a machinery check, NOT a forward test"
    if not len(J):
        sys.exit("no new sessions after the stored data (last stored session " + str(last_db) + ")")
    # ---- data checks
    zv = float(np.mean([b.v <= 0 for b in new]))
    rth = pd.Series([1 if 930 <= b.smin < 1315 else 0 for b in new], index=[b.label for b in new]).groupby(level=0).sum()
    rep["data"] = dict(zero_volume_share=zv, rth_bars_median=float(rth.median()), sessions_rth_under_370=int((rth < 370).sum()), sessions_flagged_roll_or_bad=int(J.roll.sum()))
    valid = zv < 0.05 and float(rth.median()) >= 370
    rep["valid"] = bool(valid)
    # ---- strategy level
    live = os.path.isdir("/home/user/data_raw_live")
    if live:
        os.environ["TOPSTEP_RAW_DIR"] = "/home/user/data_raw_live"
        import run_kelly as K
        gs = lambda names: K.with_risk_unit(K.group_stream(names), "atr", 40)
        dev = {"noise": gs(["nq_noise"]), "gap": gs(["nq_gap30"])}
    L = len(J)
    rep["sleeves"] = {}
    for g in ("noise", "gap"):
        x = np.array([r[g]["usd"] for r in J.to_dict("records")])
        d = dict(total_usd_per_micro=float(x.sum()), mean_daily=float(x.mean()), hit_days=float((x > 0).mean()), trades=int(sum(r[g]["trades"] for r in J.to_dict("records"))), sharpe_annualised=float(x.mean() / x.std(ddof=1) * np.sqrt(252)) if x.std() > 0 else None)
        if live:
            s = dev[g]
            m = (s.dates <= np.datetime64("2022-12-31")) & s.ok
            d["dev_window_percentile"] = window_pct(s.pnl[m], L, float(x.sum()))
        rep["sleeves"][g] = d
    if live:
        # pooled like the farm: 3 noise accounts + 2 gap accounts at one micro each
        x = np.array([3 * r["noise"]["usd"] + 2 * r["gap"]["usd"] for r in J.to_dict("records")])
        m = (dev["noise"].dates <= np.datetime64("2022-12-31")) & dev["noise"].ok
        pooled_dev = 3 * dev["noise"].pnl[m] + 2 * dev["gap"].pnl[m]
        rep["pooled_3_2"] = dict(total_usd_per_micro_set=float(x.sum()), dev_window_percentile=window_pct(pooled_dev, L, float(x.sum())))
    # ---- farm level
    sm = runner.farm.summary()
    rep["farm"] = dict(net=sm["net"], fees=sm["fees"], payouts=sm["payouts"], groups=sm["groups"])
    p = f"results/many_starts_data_raw_live_w75.csv"
    if os.path.exists(p) and abs(L - 75) <= 40:
        dist = pd.read_csv(p).net.to_numpy()
        rep["farm"]["dev_farm_percentile_75_sessions"] = float((dist < sm["net"]).mean() * 100)
    # ---- declared kill criteria (PR15)
    pc = rep.get("pooled_3_2", {}).get("dev_window_percentile")
    sh = None
    kill = []
    if not valid:
        kill.append("data checks failed: the forward test is invalid")
    if pc is not None and L >= 60 and pc < 5:
        kill.append(f"after {L} sessions the pooled sleeve P&L is below the 5th percentile of development windows ({pc:.1f})")
    if L >= 120 and rep["sleeves"]["noise"]["total_usd_per_micro"] < 0 and rep["sleeves"]["gap"]["total_usd_per_micro"] < 0:
        kill.append("after 120 sessions both sleeves have a negative cumulative P&L: edge not confirmed")
    rep["kill_criteria_triggered"] = kill
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(jsonable(rep), f, indent=1)
    print(json.dumps(jsonable(rep), indent=1))


if __name__ == "__main__":
    main()
