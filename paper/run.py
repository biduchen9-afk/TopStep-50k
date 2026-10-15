"""Command line: run the paper-trading harness and serve the live dashboard.

  Replay (stand-in feed, works offline with the stored Databento data):
      python -m paper.run --source replay --from 2026-04-01 --to 2026-06-30 --speed 600 --port 8765
  Live / forward test from the London Strategic Edge API (needs LSE_API_KEY in the environment; see paper/README.md):
      LSE_API_KEY=... python -m paper.run --source lse --symbol <SYMBOL> --start 2026-07-02 --port 8765

State is kept in --state (default paper_state.pkl, ignored by git) after every completed session; restarting resumes from it.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import numpy as np
import pandas as pd

from paper.farm import PaperFarm
from paper.feed import LSEBars, ReplayFeed
from paper.history import History
from paper.runner import PaperRunner
from paper.server import jsonable, serve


def load_costs_and_sessions():
    from market_data import build_sessions
    from pricing_config import load_config
    from real_engine import costs_from_config
    sess = build_sessions("nq")
    return sess, costs_from_config(load_config()).points(sess)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["replay", "lse"], default="replay")
    ap.add_argument("--from", dest="from_", default="2026-04-01", help="replay: first session label")
    ap.add_argument("--to", default=None, help="replay: last session label")
    ap.add_argument("--speed", type=float, default=600.0, help="replay: bars per second (0 = as fast as possible)")
    ap.add_argument("--symbol", default=None, help="lse: instrument symbol exactly as in the vault catalog (see python -m paper.lse_probe)")
    ap.add_argument("--dataset", default=None, help="lse: dataset name when the symbol exists in more than one")
    ap.add_argument("--start", default=None, help="lse: first UTC time to fetch (default: right after the last completed session)")
    ap.add_argument("--poll", type=float, default=20.0)
    ap.add_argument("--state", default="paper_state.pkl")
    ap.add_argument("--fresh", action="store_true", help="ignore an existing state file")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-server", action="store_true")
    ap.add_argument("--dump", default=None, help="write the final snapshot JSON here when the feed ends")
    ap.add_argument("--stop-at", default=None, help="replay: stop feeding at this Eastern wall-clock time (e.g. '2026-06-26 14:20') and leave that session open, to look at a mid-session state")
    a = ap.parse_args(argv)

    sess, cost_pts = load_costs_and_sessions()
    meta = dict(mode=a.source.upper(), symbol=a.symbol or "NQ (Databento replay)", strategy="3 noise-area + 2 gap-VWAP accounts, balanced sizing (the baseline)")
    if os.path.exists(a.state) and not a.fresh:
        runner = PaperRunner.load(a.state)
        runner.meta = meta
        print(f"resumed {a.state}: last completed session {runner.last_label}, farm net {runner.farm.net:+,.0f}", flush=True)
    elif a.source == "replay":
        i0 = int(np.searchsorted(sess.dates.astype("datetime64[D]"), np.datetime64(pd.Timestamp(a.from_).date())))
        runner = PaperRunner(History.from_sessions(sess, i0), PaperFarm(), cost_pts, a.state, meta, t0=i0)
        if i0 > 0:
            runner.last_label = pd.Timestamp(sess.dates[i0 - 1]).date()
    else:
        # forward test / live: warm up the history with every stored Databento session, start the farm fresh on the first new session
        n = len(sess)
        runner = PaperRunner(History.from_sessions(sess), PaperFarm(), cost_pts, a.state, meta, t0=n)
        runner.last_label = pd.Timestamp(sess.dates[n - 1]).date()
    runner.state_path = a.state
    if a.source == "replay":
        feed = ReplayFeed(sess, a.from_, a.to, speed=a.speed or None)
    else:
        if not a.symbol:
            sys.exit("--symbol is required for --source lse (find it with: python -m paper.lse_probe NQ)")
        nxt = runner.next_label()
        start = a.start or (pd.Timestamp(nxt) - pd.Timedelta(days=1)).strftime("%Y-%m-%dT18:00:00")     # wall clock; converted below
        if not a.start:
            start = pd.Timestamp(start).tz_localize("America/New_York").tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S")
        feed = LSEBars(a.symbol, start, dataset=a.dataset, poll=a.poll)

    def pump():
        try:
            stop = pd.Timestamp(a.stop_at).tz_localize("America/New_York") if a.stop_at else None
            for bar in feed:
                if stop is not None and bar.ts >= stop:
                    break
                runner.process(bar)
            if stop is None:
                runner.close_open_session()
            print("feed finished", flush=True)
            if a.dump:
                import json
                with open(a.dump, "w") as f:
                    json.dump(jsonable(runner.snapshot()), f)
        except Exception as e:                           # keep the dashboard up and show the problem
            runner.meta["error"] = f"{type(e).__name__}: {e}"
            print("feed stopped:", runner.meta["error"], flush=True)

    th = threading.Thread(target=pump, daemon=True)
    th.start()
    if a.no_server:
        th.join()
        return
    print(f"dashboard: http://{a.host}:{a.port}/", flush=True)
    serve(runner, a.host, a.port, block=True)


if __name__ == "__main__":
    main()
