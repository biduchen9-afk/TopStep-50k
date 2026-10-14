"""Pre-registration 8: the 144 pre-declared trials from the user's list (see docs/PREREGISTRATION_8.md), regular trading hours only."""
from __future__ import annotations
import itertools, math, sys
from statistics import NormalDist
import numpy as np, pandas as pd
import intraday_strategies as st
import portfolio as pf
from daily_features import build_daily
from market_data import build_sessions
from ml_meta import design, meta_label
from portfolio import DEV_END, Sleeve, basket, make_sleeve, r_series, sharpe
from run_corrected import NEW

pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 60)
C = st.CASH_OPEN
ND = NormalDist()
EULER = 0.5772156649


def trials():
    T = []
    A = [("a1", dict(or_min=15)), ("a1", dict(or_min=30)), ("a2", dict(rm=1.0)), ("a2", dict(rm=1.5)), ("a3", {}), ("a4", dict(stop=0)), ("a4", dict(stop=1)), ("a5", dict(stop=0)), ("a5", dict(stop=1)),
         ("a8", dict(gate=0)), ("a8", dict(gate=1)), ("a9", dict(mult=1.5)), ("a9", dict(mult=1.0)), ("a10", {})]
    B = [(f"b{k}", {}) for k in (1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 16, 19, 20, 23, 25, 26, 27, 36, 37, 38, 39)]
    for m in ("nq", "es", "gc"):
        for nm, pr in A + B:
            T.append(dict(market=m, idea=nm, kind="ex", params=pr))
    for nm in ("a6", "a7v", "a7n"):
        T.append(dict(market="nq", idea=nm, kind="ex", params={}))
    for base in ("noise", "vwap"):
        for g in ("adx20", "adx25", "er35", "sma200", "atrlow", "atrhigh", "hurst", "vr", "equity"):
            T.append(dict(market="nq", idea=f"gate_{g}_{base}", kind="gate", params=dict(base=base, gate=g)))
    for base in ("noise", "vwap", "gap30"):
        for mdl in ("logit", "lasso", "gbm"):
            T.append(dict(market="nq", idea=f"ml_{mdl}_{base}", kind="ml", params=dict(base=base, model=mdl, thr=0.5)))
    for base in ("noise", "vwap", "gap30"):
        T.append(dict(market="nq", idea=f"ml_gbm55_{base}", kind="ml", params=dict(base=base, model="gbm", thr=0.55)))
    return T


def base_sleeve(kind):
    if kind == "noise":
        return make_sleeve("nq", "noise_area", {}, costs=NEW, name="nq_noise")
    if kind == "vwap":
        return make_sleeve("nq", "vwap_atr", dict(k=0.3, s=0.2, start_min=C), costs=NEW, name="nq_vwap")
    return make_sleeve("nq", "vv_filt", dict(filt="gap30", k=0.3, s=0.2), costs=NEW, name="nq_gap30")


def masked(base, mask, tag):
    return Sleeve(base.name + "|" + tag, base.dates, np.where(mask, base.pnl, 0.0), np.where(mask, base.low, 0.0), base.atr, base.ok, base.volratio)


def yearly_pos(r, ok, dates, dev):
    y = pd.Series(r, index=pd.DatetimeIndex(dates))[ok & dev]
    yr = y.groupby(y.index.year).apply(lambda g: sharpe(g.to_numpy()) if len(g) > 20 else np.nan).dropna()
    return float((yr > 0).mean())


def boot_p(x, n_boot=2000, mean_block=10, seed=0):
    rng = np.random.default_rng(seed); T = len(x); mu = x.mean()
    if T < 100 or x.std() == 0:
        return 1.0
    xc = x - mu; p = 1.0 / mean_block; cnt = 0
    for _ in range(n_boot // 100):
        starts = rng.integers(0, T, size=(100, T)); cont = rng.random((100, T)) > p
        idx = starts.copy()
        for t in range(1, T):
            idx[:, t] = np.where(cont[:, t], (idx[:, t - 1] + 1) % T, starts[:, t])
        cnt += int((xc[idx].mean(axis=1) >= mu).sum())
    return (cnt + 1) / (n_boot + 1)


def main() -> None:
    import os, pickle
    T = trials(); assert len(T) == 144, len(T)
    F_nq = None; rows = []; series = {}
    CK = '/home/user/data_raw/ext_checkpoint.pkl'
    if os.path.exists(CK):
        rows, series = pickle.load(open(CK, 'rb'))
        print('resuming after', len(rows), 'trials')
    base_cache = {}
    sess_nq = build_sessions("nq")
    for k, t in enumerate(T):
        if k < len(rows):
            continue
        st.TRACE = []
        m, nm, kind, pr = t["market"], t["idea"], t["kind"], t["params"]
        if kind == "ex":
            sl = make_sleeve(m, "ex_" + nm, pr, None, costs=NEW, name=f"{m}_{nm}_{k}")
        elif kind == "gate":
            bs = base_cache.setdefault(pr["base"], base_sleeve(pr["base"]))
            Fd = pf._EXT.get("nq")
            if Fd is None:
                make_sleeve("nq", "ex_a3", {}, None, costs=NEW, name="warm"); Fd = pf._EXT["nq"]
            F = Fd[0]; g = pr["gate"]; n = len(bs.dates)
            assert np.array_equal(bs.dates, sess_nq.dates.astype("datetime64[D]"))
            if g == "adx20": mask = F["adx"] > 20
            elif g == "adx25": mask = F["adx"] > 25
            elif g == "er35": mask = F["er20"] > 0.35
            elif g == "sma200": mask = np.isfinite(F["sma200"])
            elif g == "atrlow": mask = F["atr_pct"] < 0.40
            elif g == "atrhigh": mask = F["atr_pct"] > 0.60
            elif g == "hurst": mask = F["hurst"] > 0.5
            elif g == "vr": mask = F["vr5"] > 1.0
            else:
                eq = pd.Series(np.cumsum(bs.pnl)); mask = (eq > eq.rolling(20).mean()).shift(1).fillna(False).to_numpy()
            mask = np.where(np.isnan(np.asarray(mask, dtype=float)), False, mask).astype(bool)
            if g == "sma200":                       # direction gate: re-run the base sleeve allowing only the trend side of the 200-day SMA
                ctx = pf._CTX["nq"]; ctx["trend200"] = np.where(np.isfinite(F["sma200"]), np.sign(F["close"] - F["sma200"]), 0.0)
                sl = make_sleeve("nq", "noise_area" if pr["base"] == "noise" else "vwap_atr", {} if pr["base"] == "noise" else dict(k=0.3, s=0.2, start_min=C), dict(trend=200), costs=NEW, name=f"nq_{nm}")
            else:
                sl = masked(bs, mask, nm)
        else:
            bs = base_cache.setdefault(pr["base"], base_sleeve(pr["base"]))
            Fd = pf._EXT.get("nq")
            if Fd is None:
                make_sleeve("nq", "ex_a3", {}, None, costs=NEW, name="warm"); Fd = pf._EXT["nq"]
            F = Fd[0]; ctx = pf._CTX["nq"]
            from portfolio import costs_from_config
            A_ = np.where(np.isfinite(bs.atr), bs.atr * 1.0, np.nan)            # $ ATR per micro; only the ratio gap/ATR matters, so use points ATR from the sleeve
            pts_atr = st.trailing_atr(ctx["ranges"], sess_nq.roll)
            gap = np.full(len(sess_nq), np.nan)
            for ii in range(len(sess_nq)):
                oo, _h, _l, _c, _v, mm = sess_nq.day(ii); aa = int(np.searchsorted(mm, C))
                if aa < len(oo):
                    gap[ii] = oo[aa] - F["close"][ii]
            X = design(bs.dates, F, pts_atr, gap, ctx["volratio"])
            sl = meta_label(bs, X, pr["model"], pr["thr"])
        tr = np.array(st.TRACE) if st.TRACE else np.zeros((0, 2)); st.TRACE = None
        b = basket([sl]); r = r_series(b); ok = b.ok; dev = b.dates <= DEV_END
        rth = bool(len(tr) == 0 or (tr[:, 0].min() >= C and tr[:, 1].max() <= 1315))
        x_dev = r[ok & dev]
        rows.append(dict(trial=k, market=m, idea=nm, kind=kind, params=str(pr), full=sharpe(r[ok]), dev=sharpe(x_dev), hold=sharpe(r[ok & ~dev]), pos_years=yearly_pos(r, ok, b.dates, dev),
                         n_trades=len(tr), rth_only=rth, T=int(len(x_dev)), skew=float(pd.Series(x_dev).skew()) if len(x_dev) > 10 else 0.0, kurt=float(pd.Series(x_dev).kurt() + 3) if len(x_dev) > 10 else 3.0))
        series[k] = x_dev
        pickle.dump((rows, series), open(CK, 'wb'))
        if k % 12 == 11:
            print(k + 1, "of 144", flush=True)
    t = pd.DataFrame(rows)
    # deflated Sharpe (Bailey & Lopez de Prado) with N = 144 trials; daily-unit Sharpes
    sr_d = np.array([(s.mean() / s.std()) if (len(s) > 20 and s.std() > 0) else 0.0 for s in series.values()])
    V = np.var(sr_d, ddof=1); N = len(t)
    sr0 = math.sqrt(V) * ((1 - EULER) * ND.inv_cdf(1 - 1 / N) + EULER * ND.inv_cdf(1 - 1 / (N * math.e)))
    dsr = []
    for sr, s, sk, ku in zip(sr_d, t["T"], t["skew"], t["kurt"]):
        den = math.sqrt(max(1e-9, 1 - sk * sr + (ku - 1) / 4 * sr ** 2))
        dsr.append(ND.cdf((sr - sr0) * math.sqrt(max(s - 1, 1)) / den))
    t["dsr"] = dsr; t["sr0_annual"] = sr0 * math.sqrt(252)
    t["boot_p"] = [boot_p(series[k]) if (t.loc[k, "dev"] > 0.2) else 1.0 for k in range(N)]
    t["accepted"] = (t.dsr >= 0.95) & (t.pos_years >= 0.6) & (t.hold > 0) & (t.boot_p < 0.05) & t.rth_only
    t["weak"] = (~t.accepted) & (t.dsr >= 0.8) & (t.pos_years >= 0.6) & (t.hold > 0) & t.rth_only
    t.to_csv("results/extended_trials.csv", index=False)
    assert t.rth_only.all(), "a trial traded outside regular hours"
    print(f"\nN={N}; noise-level Sharpe for the best of N (SR0) = {sr0 * math.sqrt(252):.2f}; cross-trial SD of Sharpe = {math.sqrt(V) * math.sqrt(252):.2f}")
    print("Top 15 by dev Sharpe:"); print(t.sort_values("dev", ascending=False).head(15)[["market", "idea", "kind", "params", "dev", "hold", "pos_years", "dsr", "boot_p", "accepted", "weak"]].round(2).to_string(index=False))
    print("\naccepted:", int(t.accepted.sum()), "| weak:", int(t.weak.sum()), "| dev>=0.5:", int((t.dev >= 0.5).sum()), "| dev>0:", int((t.dev > 0).sum()), "of", N)
    print("\nby kind (median dev / hold):"); print(t.groupby("kind")[["dev", "hold"]].median().round(2).to_string())


if __name__ == "__main__":
    main()
