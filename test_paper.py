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


def test_live_guards_incomplete_session_and_intersession_roll():
    """Real-feed guards: holes in regular hours keep a session out of the ATR history (an exchange early close is complete); a front-month splice at the session open inside the roll window stops trading that session."""
    import numpy as np, pandas as pd
    from paper.farm import PaperFarm
    from paper.feed import Bar
    from paper.history import History
    from paper.runner import PaperRunner
    inc = PaperRunner._incomplete
    assert inc(list(range(930, 1315))) is False or inc(list(range(930, 1315))) == False       # full day
    assert not inc(list(range(930, 1140)))                                                      # early close at 13:00 ET, no holes
    assert inc(list(range(930, 1140, 2)))                                                       # early close but half the minutes missing
    assert inc(list(range(930, 1315, 2))) and inc(list(range(930, 1000)))                       # holes / data stops early
    # roll splice: previous close 30000, session opens 300 points higher in the September roll window (2026-09-14 is a Monday inside the window)
    h = History()
    for i in range(20):
        h.add_session(300.0, 30000.0, np.full(12, 0.002), False)
    r = PaperRunner(h, PaperFarm(), 0.86, None, {}, t0=100, live_guards=True)
    r.last_label = pd.Timestamp("2026-09-11").date()
    ts = pd.Timestamp("2026-09-13 18:00", tz="America/New_York")
    for k in range(5):
        r.process(Bar(ts + pd.Timedelta(minutes=k), 30300.0, 30301.0, 30299.0, 30300.5, 10.0))
    assert r.roll_today and "roll" in r.roll_note
    r2 = PaperRunner(h, PaperFarm(), 0.86, None, {}, t0=100, live_guards=False)
    r2.last_label = pd.Timestamp("2026-09-11").date()
    for k in range(5):
        r2.process(Bar(ts + pd.Timedelta(minutes=k), 30300.0, 30301.0, 30299.0, 30300.5, 10.0))
    assert not r2.roll_today                                                                    # off by default: the parity replays are unchanged


def test_warm_up_from_bars_equals_history_from_sessions():
    """A history built bar by bar from an empty start (the live warm-up, no stored data) gives the same ATR / sigma / previous close as the stored-session history."""
    import numpy as np, pandas as pd
    from market_data import build_sessions
    from paper.feed import ReplayFeed
    from paper.history import History
    from paper.make_warmup import warm_from_bars
    sess = build_sessions("nq")
    i0 = int(np.searchsorted(sess.dates.astype("datetime64[D]"), np.datetime64("2024-01-02")))
    feed = ReplayFeed(sess, str(pd.Timestamp(sess.dates[i0 - 30]).date()), str(pd.Timestamp(sess.dates[i0 - 1]).date()), speed=None)
    r = warm_from_bars(list(feed))
    ref = History.from_sessions(sess, i0).context()
    got = r.hist.trimmed(60).context()
    assert abs(got["atr"] - ref["atr"]) < 1e-9 and abs(got["prev_close"] - ref["prev_close"]) < 1e-9
    assert np.allclose(got["sigma"], ref["sigma"], equal_nan=True)


import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------------------------ tick feed (paper/tickfeed.py)
class _Tick:
    def __init__(self, ts, price, size, replay=False):
        self.symbol, self.price, self.timestamp, self.replay, self.size = "NQ.F", price, ts, replay, size


def _ticks_from_bars(bars):
    out = []
    for b in bars:
        t0 = b.ts.tz_convert("UTC")
        for k, p in enumerate((b.o, b.h, b.l, b.c)):
            out.append(_Tick((t0 + pd.Timedelta(seconds=5 + 15 * k)).isoformat(), p, b.v / 4.0))
    return out


def _some_bars(n=150):
    import numpy as np
    from market_data import build_sessions
    from paper.feed import ReplayFeed
    sess = build_sessions("nq")
    i = int(np.searchsorted(sess.dates.astype("datetime64[D]"), np.datetime64("2024-01-09")))
    return [b for _, b in zip(range(n), ReplayFeed(sess, str(pd.Timestamp(sess.dates[i]).date()), str(pd.Timestamp(sess.dates[i]).date()), speed=None))]


def test_tick_feed_rebuilds_the_bars_exactly():
    from paper.tickfeed import TickBars
    bars = _some_bars(120); ticks = _ticks_from_bars(bars)
    tf = TickBars(lambda start: iter([t for t in ticks if start is None or pd.Timestamp(t.timestamp) >= pd.Timestamp(start, tz="UTC")]), settle=0.2, backoff=0.0, max_reconnects=0)
    got = list(tf)
    assert len(got) == len(bars)
    for a, b in zip(got, bars):
        assert a.ts == b.ts and abs(a.o - b.o) < 1e-9 and abs(a.h - b.h) < 1e-9 and abs(a.l - b.l) < 1e-9 and abs(a.c - b.c) < 1e-9 and abs(a.v - b.v) < 1e-6
    assert tf.volume_basis == "size"


def test_tick_feed_reconnects_without_double_counting():
    from paper.tickfeed import TickBars
    bars = _some_bars(60); ticks = _ticks_from_bars(bars)
    calls = []
    def connect(start):
        calls.append(start)
        sel = [t for t in ticks if start is None or pd.Timestamp(t.timestamp) >= pd.Timestamp(start, tz="UTC")]
        if len(calls) == 1:                                   # the first connection drops in the middle of minute 31
            cut = 4 * 30 + 2
            def g():
                yield from sel[:cut]
                raise ConnectionError("dropped")
            return g()
        return iter(sel)
    tf = TickBars(connect, settle=0.2, backoff=0.0, sleep=lambda s: None, max_reconnects=1)
    got = list(tf)
    assert len(calls) == 2 and calls[1] is not None
    assert [b.ts for b in got] == [b.ts for b in bars]
    assert all(abs(a.v - b.v) < 1e-6 and abs(a.h - b.h) < 1e-9 for a, b in zip(got, bars))
    assert tf.reconnects >= 1


def test_tick_feed_flushes_the_last_minute_when_ticks_stop():
    import threading, time
    from paper.tickfeed import TickBars
    now = pd.Timestamp.now(tz="UTC").floor("min") - pd.Timedelta(minutes=3)
    block = threading.Event()
    def connect(start):
        def g():
            yield _Tick((now + pd.Timedelta(seconds=10)).isoformat(), 100.0, 1.0)
            yield _Tick((now + pd.Timedelta(seconds=40)).isoformat(), 101.0, 2.0)
            block.wait(10)                                    # the stream goes quiet
        return g()
    tf = TickBars(connect, settle=0.6, max_idle=3.0, max_reconnects=0)
    t0 = time.time(); got = []
    for b in tf:
        got.append(b); block.set()
        break
    assert got and got[0].h == 101.0 and got[0].v == 3.0 and time.time() - t0 < 5


def test_runner_on_tick_built_bars_equals_runner_on_the_original_bars():
    """End to end: bars -> ticks -> TickBars -> PaperRunner gives the same session P&L as feeding the original bars."""
    import numpy as np
    from market_data import build_sessions
    from paper.farm import PaperFarm
    from paper.feed import ReplayFeed
    from paper.history import History
    from paper.runner import PaperRunner
    from paper.tickfeed import TickBars
    sess = build_sessions("nq")
    i0 = int(np.searchsorted(sess.dates.astype("datetime64[D]"), np.datetime64("2024-01-09")))
    lab = lambda i: str(pd.Timestamp(sess.dates[i]).date())
    bars = list(ReplayFeed(sess, lab(i0), lab(i0 + 2), speed=None))
    ticks = _ticks_from_bars(bars)
    def mk():
        r = PaperRunner(History.from_sessions(sess, i0), PaperFarm(), 0.86, None, {}, t0=i0, live_guards=False)
        r.last_label = pd.Timestamp(sess.dates[i0 - 1]).date()
        return r
    a, b = mk(), mk()
    for bar in bars:
        a.process(bar)
    a.close_open_session()
    tf = TickBars(lambda start: iter([t for t in ticks if start is None or pd.Timestamp(t.timestamp) >= pd.Timestamp(start, tz="UTC")]), settle=0.2, backoff=0.0, max_reconnects=0)
    for bar in tf:
        b.process(bar)
    b.close_open_session()
    ja, jb = a.journal, b.journal
    assert len(ja) == len(jb) == 3
    for x, y in zip(ja, jb):
        assert x["date"] == y["date"] and abs(x["noise"]["usd"] - y["noise"]["usd"]) < 1e-6 and abs(x["gap"]["usd"] - y["gap"]["usd"]) < 1e-6 and x["noise"]["trades"] == y["noise"]["trades"]


def test_publisher_posts_the_snapshot_with_a_bearer_token_and_never_leaks_it(tmp_path):
    import http.server
    import json
    import threading
    import pandas as pd
    from market_data import build_sessions
    from paper.farm import PaperFarm
    from paper.feed import ReplayFeed
    from paper.history import History
    from paper.publish import Publisher
    from paper.runner import PaperRunner
    from paper.server import jsonable
    from pricing_config import load_config
    from real_engine import costs_from_config
    sess = build_sessions("nq")
    cost = costs_from_config(load_config()).points(sess)
    dates = sess.dates.astype("datetime64[D]")
    i0 = int(np.searchsorted(dates, np.datetime64("2019-03-01")))
    r = PaperRunner(History.from_sessions(sess, i0), PaperFarm(), cost, None, dict(mode="REPLAY", symbol="x"), t0=i0)
    r.last_label = pd.Timestamp(sess.dates[i0 - 1]).date()
    for b in ReplayFeed(sess, pd.Timestamp(sess.dates[i0]), pd.Timestamp(sess.dates[i0 + 6])):
        r.process(b)
    seen, status = [], [200]

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            seen.append((self.path, self.headers.get("Authorization"), self.headers.get("Content-Type"), body, self.headers.get("x-vercel-protection-bypass")))
            self.send_response(status[0]); self.send_header("Content-Length", "0"); self.end_headers()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    tok = "T" * 43
    pub = Publisher(r, f"http://127.0.0.1:{srv.server_address[1]}/api/ingest", tok)
    assert pub.once() and pub.ok == 1
    path, auth, ctype, body, byp = seen[0]
    assert byp is None                                                   # no bypass header unless one is configured
    assert path == "/api/ingest" and auth == "Bearer " + tok and ctype == "application/json"
    assert tok.encode() not in body                                      # the token is not part of the published data
    got, want = json.loads(body), json.loads(json.dumps(jsonable(r.snapshot())))
    got.pop("wall"), want.pop("wall")
    assert got == want and got["farm"] is not None and len(got["journal"]) == 6        # the 7th session is still open
    pub2 = Publisher(r, pub.url, tok, bypass="B" * 32)
    assert pub2.once() and seen[-1][4] == "B" * 32 and b"B" * 32 not in seen[-1][3]    # the bypass secret goes in a header only
    status[0] = 401
    assert not pub.once() and "token rejected" in pub.last_error         # a rejected token is reported, never raised
    status[0] = 200
    srv.shutdown()
    assert not pub.once() and pub.last_error                              # server gone: reported, not raised
    with pytest.raises(ValueError):
        Publisher(r, "http://example.com/api/ingest", tok)               # plain http is only allowed for localhost


def test_remote_viewer_page_is_the_same_file_as_the_local_dashboard():
    """public/index.html is a copy of paper/static/dashboard.html (the Vercel viewer serves the very same page); edit the source and copy it over."""
    import pathlib
    root = pathlib.Path(__file__).parent
    assert (root / "public/index.html").read_bytes() == (root / "paper/static/dashboard.html").read_bytes()


# ------------------------------------------------------------------------------------------------ council 2 ops findings: regression tests for the fixes
def test_incomplete_accepts_the_1300_and_1315_early_closes_but_not_outages():
    from paper.runner import PaperRunner
    inc = PaperRunner._incomplete
    assert not inc(list(range(930, 1140)))            # 13:00 ET close (Labor Day, 3 July when it is an early close)
    assert not inc(list(range(930, 1155)))            # 13:15 ET close (day after Thanksgiving, 24 December): 34 perfect Databento sessions
    assert inc(list(range(930, 1155, 2)))             # same close, half the minutes missing
    assert inc(list(range(930, 1030)))                # outage that ends at 11:10 ET
    assert inc(list(range(930, 1100)) + list(range(1130, 1155)))   # a 30-minute hole inside an early-close day


def _flat_bars(start, n, price=30000.0):
    from paper.feed import Bar
    return [Bar(start + pd.Timedelta(minutes=k), price, price + 1.0, price - 1.0, price, 10.0) for k in range(n)]


def _quiet_runner():
    from paper.farm import PaperFarm
    from paper.history import History
    from paper.runner import PaperRunner
    h = History()
    for i in range(20):
        h.add_session(300.0, 30000.0, np.full(12, 0.002), False)
    r = PaperRunner(h, PaperFarm(), 0.86, None, {}, t0=100, live_guards=True)
    r.last_label = pd.Timestamp("2026-10-13").date()
    return r


def test_session_closes_at_16h_and_duplicate_bars_are_ignored():
    r = _quiet_runner()
    start = pd.Timestamp("2026-10-13 18:00", tz="America/New_York")             # session of Wednesday 14 October
    bars = _flat_bars(start, 1320)                                              # 18:00 to 15:59 ET
    for b in bars[:700]:
        r.process(b)
    r.process(bars[699])                                                         # duplicate of the last bar
    r.process(bars[650])                                                         # out-of-order bar
    assert len(r.bars) == 700
    for b in bars[700:]:
        r.process(b)
    assert len(r.journal) == 0 and r.cur_label is not None                       # still open at 15:59
    r.process(_flat_bars(start + pd.Timedelta(minutes=1320), 1)[0])              # the 16:00 ET bar closes the session
    assert len(r.journal) == 1 and r.cur_label is None                           # no waiting for Thursday evening's first bar


def test_tick_feed_market_hours_and_bad_ticks():
    from paper.tickfeed import TickBars
    mo = lambda s: TickBars.market_open(pd.Timestamp(s, tz="America/New_York").tz_convert("UTC"))
    assert mo("2026-10-14 11:00") and mo("2026-10-18 18:30") and mo("2026-10-16 16:30")
    assert not mo("2026-10-17 12:00") and not mo("2026-10-18 17:30") and not mo("2026-10-16 17:30") and not mo("2026-10-13 17:30")     # Saturday, Sunday before the open, Friday after the close, daily halt
    tf = TickBars(lambda s: iter([]))
    for bad in (float("nan"), -5.0, 0.0, float("inf")):
        with pytest.raises(ValueError):
            tf._parse(_Tick("2026-10-14T15:00:10+00:00", bad, 1.0))
    t, p, size, rep = tf._parse(_Tick("2026-10-14T15:00:10+00:00", 100.0, float("nan")))
    assert size == 0.0 and tf.n_bad_size == 1 and tf.n_bad_price == 4            # a NaN size counts the tick, not the contracts


def test_tick_feed_drops_a_future_stamped_tick_and_keeps_the_real_ones():
    from paper.tickfeed import TickBars
    now = pd.Timestamp("2026-10-14 15:00:55", tz="UTC")
    ticks = [_Tick("2026-10-14T15:00:10+00:00", 100.0, 1.0), _Tick("2026-10-14T15:45:00+00:00", 999.0, 1.0), _Tick("2026-10-14T15:00:30+00:00", 102.0, 1.0)]
    tf = TickBars(lambda s: iter(ticks), settle=0.2, backoff=0.0, clock=lambda: now.timestamp(), max_reconnects=0)
    got = list(tf)
    assert tf.n_future == 1 and len(got) == 1 and got[0].ts.tz_convert("UTC") == pd.Timestamp("2026-10-14 15:00", tz="UTC") and got[0].h == 102.0 and got[0].v == 2.0


def test_tick_feed_replay_pause_does_not_cut_a_minute():
    import threading, time
    from paper.tickfeed import TickBars
    old = pd.Timestamp.now(tz="UTC").floor("min") - pd.Timedelta(hours=2)
    def connect(start):
        def g():
            yield _Tick((old + pd.Timedelta(seconds=5)).isoformat(), 100.0, 3.0, replay=True)
            time.sleep(1.2)                                                      # the vendor pauses the replay for longer than the settle time
            yield _Tick((old + pd.Timedelta(seconds=50)).isoformat(), 101.0, 4.0, replay=True)
            yield _Tick((old + pd.Timedelta(seconds=70)).isoformat(), 102.0, 1.0, replay=True)
        return g()
    got = list(TickBars(connect, settle=0.3, backoff=0.0, max_reconnects=0))
    assert got[0].v == 7.0 and got[0].h == 101.0                                 # both halves of the minute are in one bar


def test_tick_feed_reconnects_when_the_socket_goes_silent():
    import threading, time
    from paper.tickfeed import TickBars
    base = pd.Timestamp("2026-10-14 15:00:00", tz="UTC").timestamp()             # Wednesday 11:00 ET, market open
    clk = [base]
    stop = threading.Event()
    def ticker():
        while not stop.is_set():
            time.sleep(0.05); clk[0] += 10.0                                     # a fast fake wall clock: 10 s per 50 ms
    threading.Thread(target=ticker, daemon=True).start()
    calls = []
    def connect(start):
        calls.append(start)
        def g():
            if len(calls) == 1:
                yield _Tick(pd.Timestamp(clk[0], unit="s", tz="UTC").isoformat(), 100.0, 1.0)
                threading.Event().wait(20)                                       # silent socket: no ticks, no error
            else:
                yield _Tick(pd.Timestamp(clk[0], unit="s", tz="UTC").isoformat(), 101.0, 1.0)
        return g()
    tf = TickBars(connect, settle=0.2, backoff=0.0, sleep=lambda s: None, clock=lambda: clk[0], stall=60.0, max_reconnects=1)
    try:
        list(tf)
    finally:
        stop.set()
    assert len(calls) == 2 and tf.n_stall >= 1 and "reconnecting" in (tf.last_error or "")


# ------------------------------------------------------------------------------------------------ PR19: verdict rules and the VWAP shadow sleeve
def _jrow(r_noise=0.0, r_gap=0.0, r_vwap=0.0, gap=False, atr=1000.0):
    return dict(date="2026-10-19", atr_usd=atr, noise=dict(usd=r_noise * atr, low=0.0, trades=1), gap=dict(usd=r_gap * atr, low=0.0, trades=1), vwap=dict(usd=r_vwap * atr, low=0.0, trades=1), data_gap=gap)


def test_pr19_verdict_kill_line_confirm_latch_and_data_rules():
    import math
    from paper.verdict import CONFIRM_R, kill_threshold, verdict19
    assert abs(kill_threshold(252) - (-3.482)) < 1e-9 and abs(kill_threshold(504) - (-1.214)) < 1e-9 and abs(kill_threshold(756) - 1.054) < 1e-9
    assert abs(CONFIRM_R - 10.887) < 0.01
    th = {str(k): -2.0 for k in range(60, 255, 5)}
    assert verdict19([_jrow() for _ in range(30)])["status"] == "TOO EARLY"
    # a steady development-sized edge continues; no kill, no confirm before 36 months
    v = verdict19([_jrow(0.0183, 0.0183) for _ in range(300)], th)
    assert v["status"] == "CONTINUE" and v["kill_session"] is None and v["next_look"] == 315 and v["pooled_R"] > 5
    # the kill line at the first look (session 63): -5.75 + 0.009 * 63 = -5.183
    j = [_jrow(-0.1, -0.1)] * 51 + [_jrow(0.0, 0.0)] * 12                          # cumulative -5.1 at session 63: not yet
    assert verdict19(j, th)["status"] != "KILL"
    j = [_jrow(-0.0825, -0.0825)] * 63                                               # -5.1975 <= -5.183: kill at 63
    v = verdict19(j, th); assert v["status"] == "KILL" and v["kill_session"] == 63
    # latched: a recovery after the kill look does not undo it
    v = verdict19(j + [_jrow(0.5, 0.5)] * 200, th); assert v["status"] == "KILL" and v["kill_session"] == 63
    # a loss that stays above the line is only a WATCH when it is below the PR18 5th percentile
    v = verdict19([_jrow(-0.04, -0.04)] * 70, th); assert v["status"] == "WATCH" and v["watch"] and v["kill_session"] is None
    # confirm needs 756 valid sessions, no kill, cumulative pooled R >= 10.89
    ok = verdict19([_jrow(0.016, 0.016)] * 756, th); assert ok["status"] == "CONFIRMED (36 months)" and ok["pooled_R"] >= CONFIRM_R
    no = verdict19([_jrow(0.012, 0.012)] * 756, th); assert no["status"] == "CANNOT TELL (36 months)"
    # data rules: gate at session 20, invalid looks, two invalid looks in a row
    v = verdict19([_jrow(gap=(i in (3, 11))) for i in range(25)]); assert v["status"] == "DATA FAILURE" and v["data_failure_session"] == 20
    jj = [_jrow(gap=(i % 30 == 5)) for i in range(63)]                              # 2 holes of 63 = 3.2% > 2%: the look at 63 is invalid (and the gate at 20 saw one hole)
    v = verdict19(jj); assert v["status"] == "INVALID" and v["invalid_looks"] == [63]
    jj = [_jrow(gap=(i % 30 == 5)) for i in range(90)]
    v = verdict19(jj); assert v["status"] == "DATA FAILURE" and v["data_failure_session"] == 84
    # shadow columns are descriptive: VWAP R and the equal-weight basket are reported, not used
    v = verdict19([_jrow(0.01, 0.02, 0.03)] * 100); assert abs(v["shadow_vwap_R"] - 3.0) < 1e-9 and abs(v["shadow_basket_R"] - 2.0) < 1e-9 and abs(v["pooled_R"] - 1.4) < 1e-9
    assert not math.isnan(v["likelihood_ratio_full_vs_zero"])


def test_runner_vwap_shadow_sleeve_matches_the_batch_sleeve_and_does_not_touch_the_farm():
    import pandas as pd
    from market_data import build_sessions
    from paper.farm import PaperFarm
    from paper.feed import ReplayFeed
    from paper.history import History
    from paper.runner import PaperRunner
    from portfolio import make_sleeve
    from pricing_config import load_config
    from real_engine import costs_from_config
    sess = build_sessions("nq"); costs = costs_from_config(load_config()); cost = costs.points(sess)
    dates = sess.dates.astype("datetime64[D]")
    i0 = int(np.searchsorted(dates, np.datetime64("2019-03-01"))); i1 = i0 + 25
    r = PaperRunner(History.from_sessions(sess, i0), PaperFarm(), cost, None, {}, t0=i0)
    r.last_label = pd.Timestamp(sess.dates[i0 - 1]).date()
    for b in ReplayFeed(sess, pd.Timestamp(sess.dates[i0]), pd.Timestamp(sess.dates[i1])):
        r.process(b)
    r.close_open_session()
    ref = make_sleeve("nq", "vwap_atr", dict(k=0.3, s=0.2, start_min=930), costs=costs, name="nq_vwap")
    pos = {str(d)[:10]: k for k, d in enumerate(ref.dates)}
    n = 0
    for row in r.journal:
        k = pos.get(row["date"])
        if k is None or not ref.ok[k] or row["roll"]:
            continue
        assert abs(row["vwap"]["usd"] - ref.pnl[k]) < 1e-6 and abs(row["vwap"]["low"] - ref.low[k]) < 1e-6, (row["date"], row["vwap"], ref.pnl[k])
        n += 1
    assert n >= 15
    f = r.snapshot()["forward"]
    assert f["status"] == "TOO EARLY" and f["sessions"] == len(r.journal)


# ------------------------------------------------------------------------------------------------ operations tools: day-0 checks, key file, supervisor
def test_day0_inspects_fields_and_compares_tick_bars_with_vendor_candles():
    from paper.day0 import compare_bars, inspect_ticks, verdict
    from paper.feed import Bar
    info = inspect_ticks([_Tick("2026-10-14T15:00:10+00:00", 100.0, 3.0), _Tick("2026-10-14T15:00:20+00:00", 100.25, 1.0, replay=True)])
    assert info["size_field"] == "size" and info["has_replay_flag"] and info["has_timestamp"] and "price" in info["fields"]
    class NoSize:
        symbol, price, timestamp, replay = "NQ.F", 1.0, "2026-10-14T15:00:10+00:00", False
    assert inspect_ticks([NoSize()])["size_field"] is None                                  # prices only: NO-GO
    t0 = pd.Timestamp("2026-10-14 11:00", tz="America/New_York")
    bars = [Bar(t0 + pd.Timedelta(minutes=k), 100.0, 101.0, 99.0, 100.0 + (k % 3) * 0.25, 10.0) for k in range(70)]
    same = compare_bars(bars, list(bars))
    assert same["minutes_both"] == 70 and same["close_match"] == 1.0 and same["volume_match"] == 1.0 and verdict(info, same)["go"]
    off = [Bar(b.ts, b.o, b.h, b.l, b.c + (2.0 if k % 5 == 0 else 0.0), b.v * (1.2 if k % 4 == 0 else 1.0)) for k, b in enumerate(bars)]
    bad = compare_bars(bars, off)
    assert abs(bad["close_match"] - (1 - 14 / 70)) < 1e-9 and bad["volume_match"] < 0.8 and not verdict(info, bad)["go"]       # 80% < 95%: NO-GO
    short = compare_bars(bars[:40], bars[:40]); assert not verdict(info, short)["go"]                                                 # fewer than 60 minutes
    assert not verdict(inspect_ticks([NoSize()]), same)["go"]


def test_key_file_is_read_without_printing_and_warns_when_world_readable(tmp_path, monkeypatch, capsys):
    import os
    from paper.run import load_key_file
    monkeypatch.delenv("LSE_API_KEY", raising=False)
    f = tmp_path / "k"; f.write_text("  secret-key-123  \n"); os.chmod(f, 0o600)
    load_key_file(str(f)); assert os.environ["LSE_API_KEY"] == "secret-key-123" and "secret" not in capsys.readouterr().out
    monkeypatch.delenv("LSE_API_KEY"); os.chmod(f, 0o644)
    load_key_file(str(f)); out = capsys.readouterr().out; assert "chmod 600" in out and "secret" not in out
    monkeypatch.setenv("LSE_API_KEY", "from-env"); load_key_file(str(f)); assert os.environ["LSE_API_KEY"] == "from-env"         # the environment wins
    monkeypatch.delenv("LSE_API_KEY"); (tmp_path / "e").write_text(" \n")
    with pytest.raises(SystemExit):
        load_key_file(str(tmp_path / "e"))


def test_supervisor_restarts_keeps_the_clock_and_backs_up(tmp_path, monkeypatch):
    import sys
    from paper import supervise as S
    st = tmp_path / "state.pkl"
    # restart arguments: drop --fresh and the warm-up only when a state file exists
    a = ["--source", "ticks", "--warm-from-lse", "60", "--fresh", "--state", str(st), "--port", "8765"]
    assert S.restart_args(a, str(st)) == a
    st.write_bytes(b"x")
    assert S.restart_args(a, str(st)) == ["--source", "ticks", "--state", str(st), "--port", "8765"]
    assert S.restart_args(["--warm-from-lse=60", "--fresh", "--x"], str(st)) == ["--x"]
    # rolling backups of the state and the bar file, newest 14 kept
    (tmp_path / "state_bars.csv").write_text("a\n")
    d = tmp_path / "bk"
    for k in range(17):
        S.backup(str(st), str(d), keep=14, now=1_700_000_000 + 3600 * k)
    names = sorted(x.name for x in d.iterdir()); assert len([n for n in names if n.startswith("state.pkl.")]) == 14 and len([n for n in names if n.startswith("state_bars.csv.")]) == 14
    # the loop: a child that exits is relaunched with backoff until max_restarts
    script = tmp_path / "fake_run.py"; script.write_text("import sys; print('args', sys.argv[1:]); sys.exit(3)\n")
    sleeps = []; calls = []
    real_popen = S.subprocess.Popen
    def popen(cmd, **kw):
        calls.append(cmd); return real_popen([sys.executable, str(script)] + cmd[3:], **kw)
    monkeypatch.setattr(S.subprocess, "Popen", popen)
    rc = S.supervise(a, log=str(tmp_path / "log.txt"), backup_dir=None, max_restarts=2, sleep=sleeps.append, poll=0.01)
    assert rc == 3 and len(calls) == 3 and sleeps[:2] == [10.0, 20.0]                         # 5 s * 2**fails, doubling
    assert "--fresh" in calls[0] and "--fresh" not in calls[1] and "--warm-from-lse" not in calls[2]
    log = (tmp_path / "log.txt").read_text(); assert "exited with code 3" in log and "launch 3" in log
