"""Pre-registration 16 evaluator: account level (EV per Combine start with year-block bootstrap) and farm level (247 overlapping 3-year dev farms).

Runs on the repaired every-session data (TOPSTEP_RAW_DIR, default /home/user/data_raw_live) and rules v2 (TOPSTEP_RULES, default v2).
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("TOPSTEP_RAW_DIR", "/home/user/data_raw_live")
import numpy as np
import pandas as pd

import academic_strategies as ac
import intraday_strategies as st
import market_data as md
from firm_engine import run_eval_firm, run_funded_firm
from multi_firm import FirmPlan, Policy, run_multifirm
from portfolio import DEV_END
from real_engine import FundedPlan, Stream

sys.argv = sys.argv[:1]
import run_kelly as K

S = K.S
CYCLE = 21
GROUPS = {"noise": ["nq_noise"], "gap": ["nq_gap30"]}
BASE = dict(a1=1.2, t1=2000.0, a2=0.8, R=9600.0)
STEP, WINDOW = 10, 756


# ---------------------------------------------------------------------------------------------- data
def load() -> dict:
    gs = lambda names: K.with_risk_unit(K.group_stream(names), "atr", 40)
    ST = {g: gs(n) for g, n in GROUPS.items()}
    ref = ST["noise"]; n = len(ref.dates); n_dev = int(np.searchsorted(ref.dates, DEV_END, side="right"))
    return dict(ST=ST, ref=ref, n=n, n_dev=n_dev, dates=pd.DatetimeIndex(ref.dates))


def features(ref: Stream) -> Dict[str, np.ndarray]:
    """Per-session pre-trade features on the same dates as the streams: |gap|/ATR, previous range/ATR, 5/60 range ratio."""
    sess = md.build_sessions("nq")
    assert np.isin(sess.dates, ref.dates).all(), "an NQ session is missing from the stream dates"
    ctx = ac.context(sess)
    atr = st.trailing_atr(ctx["ranges"], sess.roll)
    n = len(sess)
    gap = np.full(n, np.nan)
    for i in range(n):
        o, h, l, c, v, m = sess.day(i)
        a = int(np.searchsorted(m, st.CASH_OPEN))
        if a < len(o) and np.isfinite(ctx["prev_close"][i]):
            gap[i] = o[a] - ctx["prev_close"][i]
    prev_rng = np.r_[np.nan, ctx["ranges"][:-1]]
    raw = dict(gap=np.abs(gap) / atr, rng=prev_rng / atr, vol=ctx["volratio"])
    pos = pd.Index(sess.dates).get_indexer(ref.dates)                  # the streams carry a couple of dates without an NQ session
    return {k: np.where(pos >= 0, v[np.maximum(pos, 0)], np.nan) for k, v in raw.items()}


def skip_mask(F: Dict[str, np.ndarray], kind: str, thr: float) -> np.ndarray:
    return np.nan_to_num(F[kind], nan=-1.0) > thr


def filtered(s: Stream, skip: np.ndarray) -> Stream:
    """No trade on skipped sessions (pnl and low become 0); everything else untouched."""
    return Stream(s.name + f"|skip{int(skip.sum())}", s.dates, np.where(skip, 0.0, s.pnl), np.where(skip, 0.0, s.low), s.atr_usd, s.ok)


# ---------------------------------------------------------------------------------------------- account level
def mk_plan(a1: float, t1: float, a2: float, w1: Optional[float] = None, scale=None, scale2=None) -> FundedPlan:
    frac = S["spec"].funded.payout_frac
    return FundedPlan("p", "x", t1, frac if w1 is None else w1, 1.0, 0.0, frac, 1.0, S["max_payouts"], a1, a2, scale, scale2)


class AcctEval:
    """Combine outcomes of one stream on sequential dev starts (computed once per R); funded runs per plan."""

    def __init__(self, s: Stream, n_dev: int, R: float, starts: Optional[List[int]] = None):
        self.spec = S["spec"]; self.s, self.n_dev, self.R = s, n_dev, R
        self.x, self.lo, self.atr = s.pnl[:n_dev].tolist(), s.low[:n_dev].tolist(), s.atr_usd[:n_dev].tolist()
        self.nd = s.desired_micros(R)[:n_dev]
        first = int(np.flatnonzero(s.ok)[0]) + 260
        self.starts = list(range(first, n_dev - 400, 3)) if starts is None else list(starts)
        self.d = pd.DatetimeIndex(s.dates)
        self.ev = [(i0,) + run_eval_firm(self.spec.eval_phases, self.x, self.lo, self.nd, i0) for i0 in self.starts]

    def run(self, a1: float, t1: float, a2: float, w1=None, scale=None, scale2=None) -> pd.DataFrame:
        plan = mk_plan(a1, t1, a2, w1, scale, scale2)
        fee = self.spec.eval_fee
        rows = []
        for i0, status, tau in self.ev:
            fees = fee * math.ceil(tau / CYCLE) + (self.spec.activation_fee if status == 1 else 0.0)
            r = dict(start=self.d[i0], passed=status == 1, paid=False, d_total=np.nan, first=np.nan, total=0.0, fees=fees, tau=tau)
            if status == 1:
                act = min(i0 + tau + 1, self.n_dev - 1); tm: list = []
                pays, *_ = run_funded_firm(self.spec.funded, self.x, self.lo, self.nd, act, plan, self.atr, tm)
                if pays:
                    r.update(paid=True, d_total=(self.d[min(tm[0], self.n_dev - 1)] - self.d[i0]).days, first=pays[0], total=sum(pays))
            rows.append(r)
        df = pd.DataFrame(rows); df["ev"] = df.total - df.fees
        return df


def summ(df: pd.DataFrame) -> dict:
    p = df[df.passed]
    return dict(starts=len(df), pass_rate=df.passed.mean(), funded=len(p), paid_share=p.paid.mean() if len(p) else np.nan,
                days_med=p[p.paid].d_total.median() if p.paid.any() else np.nan, within_90=(p.d_total <= 90).mean() if len(p) else np.nan,
                payouts_per_funded=p.total.mean() if len(p) else np.nan, fees_per_start=df.fees.mean(), ev=df.ev.mean(), rate=df.ev.sum() / df.tau.sum(), mean_tau=df.tau.mean(),
                fees_per_funded=df.fees.sum() / max(len(p), 1))


def halves_rate(df: pd.DataFrame) -> Tuple[float, float]:
    a, b = df[df.start <= "2016-12-31"], df[df.start > "2016-12-31"]
    return float(a.ev.sum() / a.tau.sum()), float(b.ev.sum() / b.tau.sum())


def boot_rate_se(a: pd.DataFrame, b: pd.DataFrame, B: int = 2000, seed: int = 16) -> float:
    """Year-block bootstrap SE of rate(a) - rate(b) (rate = sum EV / sum Combine sessions); a and b are on the same starts."""
    rng = np.random.default_rng(seed)
    years = a.start.dt.year.values; ys = np.unique(years)
    ea = np.array([a.ev.values[years == y].sum() for y in ys]); ta = np.array([a.tau.values[years == y].sum() for y in ys])
    eb = np.array([b.ev.values[years == y].sum() for y in ys]); tb = np.array([b.tau.values[years == y].sum() for y in ys])
    k = len(ys); idx = rng.integers(0, k, size=(B, k))
    return float((ea[idx].sum(1) / ta[idx].sum(1) - eb[idx].sum(1) / tb[idx].sum(1)).std())


def time_shares(w_start: Dict[str, float], mean_tau: Dict[str, float]) -> Dict[str, float]:
    """Share of Combine sessions per group in a one-at-a-time farm: starts share x mean Combine length."""
    z = sum(w_start[g] * mean_tau[g] for g in w_start)
    return {g: w_start[g] * mean_tau[g] / z for g in w_start}


# ---------------------------------------------------------------------------------------------- farm level
def farm_plans(ST: Dict[str, Stream], groups: List[Tuple[str, int, dict]]) -> List[FirmPlan]:
    """groups = [(group name, accounts, dict(a1, t1, a2, R[, w1, scale, scale2, eval_scale]))]."""
    out = []
    for g, k, p in groups:
        out.append(FirmPlan(replace(S["spec"], name=g, max_funded=k, max_evals=1), [ST[g]], p.get("R", BASE["R"]), mk_plan(p["a1"], p["t1"], p["a2"], p.get("w1"), p.get("scale"), p.get("scale2")),
                            p_est=1.0, refill="all", eval_scale=p.get("eval_scale")))
    return out


def farm_starts(ref: Stream, n_dev: int) -> List[int]:
    first = int(np.flatnonzero(ref.ok)[0])
    return list(range(first, n_dev - WINDOW, STEP))


def farm_stats(fps: List[FirmPlan], ref: Stream, n_dev: int, starts: Optional[List[int]] = None, seen: bool = True) -> dict:
    pol = Policy(reserve_cycles=None, global_one_eval=True)
    starts = farm_starts(ref, n_dev) if starts is None else starts
    nets, evals, funded, paid, d1 = [], {}, {}, {}, []
    dts = pd.DatetimeIndex(ref.dates)
    for i0 in starts:
        r = run_multifirm(fps, pol, i0, i0 + WINDOW); nets.append(r["net"])
        fb = [v["first_buy_t"] for v in r["firms"].values() if v["first_buy_t"] is not None]; fp_ = [v["first_payout_t"] for v in r["firms"].values() if v["first_payout_t"] is not None]
        if fb and fp_:
            d1.append((dts[min(min(fp_), len(dts) - 1)] - dts[min(fb)]).days)
        for g, v in r["firms"].items():
            evals[g] = evals.get(g, 0) + v["n_evals"]; funded[g] = funded.get(g, 0) + v["n_funded"]; paid[g] = paid.get(g, 0) + v["n_paid_accts"]
    nets = np.array(nets); q = np.percentile(nets, [10, 25, 50, 75, 90])
    out = dict(farms=len(nets), mean=nets.mean(), p10=q[0], p25=q[1], median=q[2], p75=q[3], p90=q[4], share_pos=float((nets > 0).mean()), worst=nets.min(),
               funded_total=sum(funded.values()), paid_share_farm=sum(paid.values()) / max(sum(funded.values()), 1), evals=evals, funded=funded, paid=paid, nets=nets,
               days_first_payout=float(np.median(d1)) if d1 else np.nan, farms_paid_in_window=len(d1) / len(nets))
    if seen:
        ix = lambda dt: int(np.searchsorted(ref.dates, np.datetime64(dt)))
        pol2 = Policy(reserve_cycles=None, global_one_eval=True)
        out["seen"] = [round(run_multifirm(fps, pol2, ix(x))["net"]) for x in ("2021-01-04", "2023-01-02", "2025-01-02", "2025-07-01")]
    return out


def farm_mix(base_stats: dict) -> Tuple[Dict[str, float], Dict[str, float]]:
    """(share of Combine starts, share of funded accounts) per group, as the baseline farm actually produces them."""
    e, f = base_stats["evals"], base_stats["funded"]
    return {g: v / sum(e.values()) for g, v in e.items()}, {g: v / sum(f.values()) for g, v in f.items()}
