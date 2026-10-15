"""Parity tests: the bar-by-bar sleeves must reproduce the vectorised back-test, session by session (this is also a no-look-ahead check, the event-driven versions can only use past bars)."""
import numpy as np
import pytest

pytest.importorskip("pandas")
from market_data import RAW_DIR  # noqa: E402

pytestmark = pytest.mark.skipif(not (RAW_DIR / "nq_sessions_v2.npz").exists(), reason="needs the Databento NQ sessions cache")


def _setup():
    import sys
    from market_data import build_sessions
    from paper.history import History
    from portfolio import make_sleeve
    from pricing_config import load_config
    from real_engine import MICRO_PER_MINI, costs_from_config
    sess = build_sessions("nq")
    costs = costs_from_config(load_config())
    cost_pts = costs.points(sess)
    ref = {"nq_noise": make_sleeve("nq", "noise_area", {}, costs=costs, name="nq_noise"),
           "nq_vwap": make_sleeve("nq", "vwap_atr", dict(k=0.3, s=0.2, start_min=930), costs=costs, name="nq_vwap"),
           "nq_gap30": make_sleeve("nq", "vv_filt", dict(filt="gap30", k=0.3, s=0.2), costs=costs, name="nq_gap30")}
    return sess, cost_pts, ref, History, MICRO_PER_MINI


def test_sleeves_match_backtest_on_a_sample_of_sessions():
    from paper.replay import run_session
    sess, cost_pts, ref, History, MPM = _setup()
    n = len(sess)
    sample = set(range(40, n, 11)) | set(range(n - 120, n))
    hist = History()
    worst = {k: 0.0 for k in ref}
    checked = 0
    for i in range(n):
        if i in sample:
            res = run_session(sess, i, hist, cost_pts=cost_pts)
            for name, (pts, low, tr) in res.items():
                pnl_ref, low_ref = ref[name].pnl[i], ref[name].low[i]
                pnl_new, low_new = pts * sess.mult / MPM, low * sess.mult / MPM
                if not ref[name].ok[i]:
                    assert tr == 0 and pts == 0.0, (name, i)
                    continue
                worst[name] = max(worst[name], abs(pnl_ref - pnl_new), abs(low_ref - low_new))
            checked += 1
        o, h, l, c, v, m = sess.day(i)
        hist.add_bars(o, h, l, c, m, bool(sess.roll[i]))
    assert checked > 100
    for name, w in worst.items():
        assert w < 1e-6, (name, w)


def test_history_context_matches_backtest_atr_and_sigma():
    import academic_strategies as ac
    import intraday_strategies as st
    from market_data import build_sessions
    from paper.history import History
    sess = build_sessions("nq")
    ctx = ac.context(sess)
    atr = st.trailing_atr(ctx["ranges"], sess.roll)
    hist = History()
    n = len(sess)
    for i in range(n):
        if i % 17 == 0 or i > n - 50:
            c = hist.context()
            if np.isnan(atr[i]):
                assert np.isnan(c["atr"])
            else:
                assert abs(atr[i] - c["atr"]) < 1e-9
            if i >= 14:
                assert np.allclose(ctx["sigma"][i], c["sigma"], equal_nan=True)
            if i >= 1:
                assert ctx["prev_close"][i] == c["prev_close"]
        o, h, l, c_, v, m = sess.day(i)
        hist.add_bars(o, h, l, c_, m, bool(sess.roll[i]))


# --------------------------------------------------------------------------------------------------------- account simulators
def _streams():
    import sys
    sys.argv = sys.argv[:1]
    import run_kelly as K
    gs = lambda names: K.with_risk_unit(K.group_stream(names), "atr", 40)
    return {"noise": gs(["nq_noise"]), "gap": gs(["nq_gap30"])}


def test_eval_and_funded_accounts_match_the_batch_engines():
    from firm_engine import run_eval_firm, run_funded_firm, topstep_150k
    from paper.accounts import EvalAccount, FundedAccount
    from real_engine import FundedPlan
    ST = _streams()
    spec = topstep_150k()
    plans = [FundedPlan("p", "x", 2000.0, 0.5, 1.0, 0.0, 0.5, 1.0, 99, 1.2, 0.8), FundedPlan("p", "x", 4000.0, 0.5, 1.0, 0.0, 0.5, 1.0, 99, 0.4, 0.8),
             FundedPlan("p", "x", 1000.0, 0.3, 1.0, 0.0, 0.5, 1.0, 99, 0.8, 0.8, ("ladder", ((500.0, 2.0),)), ("npay", ((2, 1.2),)))]
    n_cmp = 0
    for g, s in ST.items():
        x, lo, atr = s.pnl.tolist(), s.low.tolist(), s.atr_usd.tolist()
        nd = s.desired_micros(9600.0)
        starts = list(range(300, len(x) - 700, 97))
        for i0 in starts:
            st, tau = run_eval_firm(spec.eval_phases, x, lo, nd, i0)
            a = EvalAccount(spec.eval_phases[0])
            t = i0
            while a.status is None and t < len(x):
                a.end_day(x[t], lo[t], a.size(nd[t]))
                t += 1
            assert (a.status, a.age) == (st, tau), (g, i0, a.status, a.age, st, tau)
            n_cmp += 1
            for plan in plans:
                tm = []
                pays, days, state, bal = run_funded_firm(spec.funded, x, lo, nd, i0, plan, atr, tm)
                f = FundedAccount(spec.funded, plan, i0)
                t = i0
                while f.state == 0 and t < len(x):
                    k = f.size(atr[t], nd[t], t)
                    f.end_day(x[t], lo[t], k, t)
                    t += 1
                assert len(f.pays) == len(pays) and np.allclose(f.pays, pays) and f.pay_times == tm, (g, i0, plan)
                assert f.state == state or (state == 0 and f.state == 0), (g, i0, f.state, state)
                n_cmp += 1
    assert n_cmp > 150


def test_paper_farm_matches_run_multifirm():
    import numpy as np
    from dataclasses import replace
    from firm_engine import topstep_150k
    from multi_firm import FirmPlan, Policy, run_multifirm
    from paper.farm import BALANCED_PLAN, run_streams
    from portfolio import DEV_END
    ST = _streams()
    ref = ST["noise"]
    n = len(ref.dates)
    n_dev = int(np.searchsorted(ref.dates, DEV_END, side="right"))
    idx = lambda dt: int(np.searchsorted(ref.dates, np.datetime64(dt)))
    spec = topstep_150k()
    fps = [FirmPlan(replace(spec, name=g, max_funded=k, max_evals=1), [ST[g]], 9600.0, BALANCED_PLAN, p_est=1.0, refill="all") for g, k in (("noise", 3), ("gap", 2))]
    pol = Policy(reserve_cycles=None, global_one_eval=True)
    cases = [(idx(f"{y}-01-01"), min(idx(f"{y}-01-01") + 756, n_dev)) for y in (2011, 2013, 2016, 2018, 2020)] + [(idx("2021-01-04"), n), (idx("2023-01-02"), n), (idx("2025-01-02"), n), (int(np.flatnonzero(ref.ok)[0]), n)]
    for i0, ne in cases:
        r = run_multifirm(fps, pol, i0, ne, log=True)
        f = run_streams(ST, i0, ne)
        assert abs(f.net - r["net"]) < 1e-6, (i0, ne, f.net, r["net"])
        assert abs(f.fees - r["fees"]) < 1e-6 and abs(f.payouts - r["payouts"]) < 1e-6, (i0, ne)
        for g in ("noise", "gap"):
            gs, rv = f.group(g), r["firms"][g]
            assert (gs.n_evals, gs.n_passed, gs.n_funded, gs.n_payouts, gs.n_paid_accts, gs.died, gs.wasted) == (rv["n_evals"], rv["n_passed"], rv["n_funded"], rv["n_payouts"], rv["n_paid_accts"], rv["died"], rv["wasted"]), (i0, g)
            assert gs.first_buy_t == rv["first_buy_t"], (i0, g)
        # multi_firm.first_payout_t is the first payout of the first-activated paying account of a group, not the earliest payout; compare the earliest payout from the event log instead
        pay_t = [e[0] for e in r["events"] if e[2] == "payout"]
        mine = [x for x in (f.group("noise").first_payout_t, f.group("gap").first_payout_t) if x is not None]
        assert (min(pay_t) if pay_t else None) == (min(mine) if mine else None), (i0, ne)


# --------------------------------------------------------------------------------------------------------------- feeds and runner
def test_lse_fetch_pages_through_a_fake_vault():
    import json, threading, urllib.parse
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import pandas as pd
    import paper.feed as feed

    rows_all = [{"ts": (pd.Timestamp("2026-07-06 22:00:00") + pd.Timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S"), "open": 100 + i, "high": 101 + i, "low": 99 + i, "close": 100.5 + i, "volume": 10 + i} for i in range(23)]
    seen_keys = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(u.query)
            seen_keys.append(self.headers.get("x-api-key"))
            assert u.path == "/vault/candles" and q["timeframe"] == ["1m"] and q["symbol"] == ["TEST"]
            start = pd.Timestamp(q["start"][0].replace("T", " ")) if "start" in q else pd.Timestamp("1970-01-01")
            lim = int(q["limit"][0])
            out = [r for r in rows_all if pd.Timestamp(r["ts"]) >= start][:lim]
            body = json.dumps(out).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    srv = HTTPServer(("127.0.0.1", 0), H)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    old = feed.VAULT_URL
    feed.VAULT_URL = f"http://127.0.0.1:{srv.server_port}/vault"
    try:
        bars = feed.lse_fetch("TEST", "2026-07-06T22:00:00", key="k-test", limit=10, pause=0.0)
    finally:
        feed.VAULT_URL = old
        srv.shutdown()
    assert len(bars) == 23 and bars[0].ts.hour == 18 and bars[0].smin == 0 and bars[0].label == pd.Timestamp("2026-07-07").date()
    assert all(b.ts < c.ts for b, c in zip(bars, bars[1:])) and set(seen_keys) == {"k-test"} and bars[-1].v == 32


def test_runner_replay_matches_batch_farm_and_resumes(tmp_path):
    """Feeding stored sessions bar by bar through PaperRunner gives the same farm as the batch run over the stored streams (same sessions, same accounts), and a saved state resumes."""
    import numpy as np
    import pandas as pd
    from market_data import build_sessions
    from paper.farm import PaperFarm
    from paper.feed import ReplayFeed
    from paper.history import History
    from paper.runner import PaperRunner
    from pricing_config import load_config
    from real_engine import costs_from_config
    sess = build_sessions("nq")
    cost = costs_from_config(load_config()).points(sess)
    dates = sess.dates.astype("datetime64[D]")
    i0 = int(np.searchsorted(dates, np.datetime64("2019-03-01")))
    i1 = int(np.searchsorted(dates, np.datetime64("2019-08-30")))
    r = PaperRunner(History.from_sessions(sess, i0), PaperFarm(), cost, str(tmp_path / "s.pkl"), {}, t0=i0)
    r.last_label = pd.Timestamp(sess.dates[i0 - 1]).date()
    mid = pd.Timestamp(sess.dates[(i0 + i1) // 2]).date()
    for b in ReplayFeed(sess, pd.Timestamp(sess.dates[i0]), mid):
        r.process(b)
    r.close_open_session()
    r2 = PaperRunner.load(str(tmp_path / "s.pkl"))              # resume from the saved state and continue
    assert r2.last_label == r.last_label and abs(r2.farm.net - r.farm.net) < 1e-9
    for b in ReplayFeed(sess, pd.Timestamp(mid) + pd.Timedelta(days=1), pd.Timestamp(sess.dates[i1])):
        r2.process(b)
    r2.close_open_session()
    # the same period straight through, from the batch streams of the three-sleeve engine
    r3 = PaperRunner(History.from_sessions(sess, i0), PaperFarm(), cost, None, {}, t0=i0)
    r3.last_label = pd.Timestamp(sess.dates[i0 - 1]).date()
    for b in ReplayFeed(sess, pd.Timestamp(sess.dates[i0]), pd.Timestamp(sess.dates[i1])):
        r3.process(b)
    r3.close_open_session()
    assert abs(r2.farm.net - r3.farm.net) < 1e-9 and abs(r2.farm.fees - r3.farm.fees) < 1e-9 and r2.farm.summary()["groups"] == r3.farm.summary()["groups"]
    assert len(r3.journal) == i1 - i0 + 1


def test_forward_verdict_logic():
    from paper.forward_test import kill_reasons, verdict
    assert verdict(False, kill_reasons(False, 10, None, 0.0, 0.0)) == "INVALID"
    assert verdict(True, kill_reasons(True, 30, 1.0, -1.0, -1.0)) == "CONTINUE"                      # too early for any rule
    assert verdict(True, kill_reasons(True, 60, 4.9, 1.0, 1.0)) == "STOP"                             # below the 5th percentile after 60 sessions
    assert verdict(True, kill_reasons(True, 60, 5.0, 1.0, 1.0)) == "CONTINUE"
    assert verdict(True, kill_reasons(True, 120, 50.0, -0.1, -0.2)) == "STOP"                         # both sleeves negative after 120
    assert verdict(True, kill_reasons(True, 120, 50.0, -0.1, 0.2)) == "CONTINUE"
    assert verdict(True, kill_reasons(True, 500, 99.0, 9.0, 9.0)) == "CONTINUE"                       # no CONFIRMED state exists
    assert verdict(True, [], reference_ok=False) == "UNKNOWN"                                         # rule (ii) could not be evaluated: never a silent CONTINUE
    assert verdict(True, ["x"], reference_ok=False) == "STOP"
