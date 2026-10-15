"""Build the small warm-up file that lets the harness start on a machine WITHOUT the 2.7 GB stored Databento history.

   python -m paper.make_warmup --bars data_new/nq_lse_1m.pkl --through 2026-10-14 --out paper_warmup.pkl

The history of every stored Databento session (through 2026-07-02) is replayed, then the LSE bars are processed by the same runner (real-feed guards on: a front-month splice at a session open and sessions with holes
in regular hours are kept out of the ATR / noise-area history), and only the last 60 sessions are kept. The farm is NOT part of the file: the farm starts fresh on the first live session.
Run `python -m paper.run --source lse --symbol NQ.F --dataset futures --warmup paper_warmup.pkl` on the machine that will run the live paper trading.
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd

from paper.farm import PaperFarm
from paper.history import History
from paper.run import load_costs_and_sessions
from paper.runner import PaperRunner


def warm_from_bars(bars, hist: History = None, last_label=None, t0: int = 0, cost_pts: float = 0.86, through=None) -> PaperRunner:
    """Process `bars` with a throw-away farm to build the ATR / noise-area history (real-feed guards on). The still-open last session is not included unless `through` says so.
    Returns the runner; use `runner.hist`, `runner.last_label`, `runner.t_next`."""
    bars = [b for b in bars if last_label is None or b.label > last_label]
    if through is None and bars:
        lt = bars[-1].ts
        through = bars[-1].label if (lt.hour * 60 + lt.minute - 18 * 60) % 1440 >= 1319 else max([b.label for b in bars if b.label < bars[-1].label], default=None)
    r = PaperRunner(hist if hist is not None else History(), PaperFarm(), cost_pts, None, dict(mode="WARMUP"), t0=t0, live_guards=True)
    r.last_label = last_label
    for b in bars:
        if through is not None and b.label <= through:
            r.process(b)
    r.close_open_session()
    return r


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", required=True)
    ap.add_argument("--through", required=True, help="last COMPLETE session label to include (the session in progress must not be included)")
    ap.add_argument("--out", default="paper_warmup.pkl")
    ap.add_argument("--keep", type=int, default=60)
    a = ap.parse_args(argv)
    bars = pickle.load(open(a.bars, "rb"))
    sess, cost_pts = load_costs_and_sessions()
    n = len(sess)
    r = PaperRunner(History.from_sessions(sess, n), PaperFarm(), cost_pts, None, dict(mode="WARMUP"), t0=n, live_guards=True)
    r.last_label = pd.Timestamp(sess.dates[n - 1]).date()
    through = pd.Timestamp(a.through).date()
    for b in bars:
        if r.last_label < b.label <= through:
            r.process(b)
    r.close_open_session()
    gaps = [j["date"] for j in r.journal if j.get("data_gap")]
    rolls = [j["date"] for j in r.journal if j.get("roll")]
    w = dict(hist=r.hist.trimmed(a.keep), t0=r.t_next, last_label=r.last_label, built_from=dict(bars=a.bars, through=a.through, sessions_with_holes=gaps, roll_or_bad=rolls))
    pickle.dump(w, open(a.out, "wb"))
    print(f"warm-up written to {a.out}: last completed session {r.last_label}, {len(w['hist'].ranges)} sessions kept, t0 {r.t_next}; sessions kept out of the ATR history: holes {len(gaps)}, roll/bad {rolls}")
    ctx = w["hist"].context()
    print(f"next-session context: ATR {ctx['atr']:.1f} points, previous close {ctx['prev_close']:.2f}")


if __name__ == "__main__":
    main()
