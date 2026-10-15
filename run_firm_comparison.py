"""Pre-registration 4, Part A: standalone farm for each firm variant on the final NQ basket. See docs/PREREGISTRATION_4.md."""
from __future__ import annotations
import itertools, pickle
import numpy as np, pandas as pd
from final_strategy import build_stream
from firm_engine import run_eval_firm, run_funded_firm
from firm_specs import specs
from multi_firm import FirmPlan, Policy, run_multifirm
from portfolio import DEV_END
from real_engine import FundedPlan, Stream

pd.set_option("display.width", 250, "display.max_columns", 40)
YEARS = list(range(2011, 2021))
RS, A1S, A2S = (6400., 9600., 12800.), (0.4, 0.8), (0.4, 0.8, 1.6)


def mkplan(meta, t1, a1, a2):
    sp = meta["spec"].funded
    return FundedPlan("p", "x", t1, sp.payout_frac, 1.0, 0.0, sp.payout_frac, 1.0, meta["max_payouts"], a1, a2)


def farm(s, meta, R, plan, i0, n_end=None):
    return run_multifirm([FirmPlan(meta["spec"], [s], R, plan)], Policy(reserve_cycles=None), i0, n_end)


def timing(s, meta, R, plan, n_end, starts):
    spec = meta["spec"]; d = pd.DatetimeIndex(s.dates)
    x, lo, atr = s.pnl[:n_end].tolist(), s.low[:n_end].tolist(), s.atr_usd[:n_end].tolist(); nd = s.desired_micros(R)[:n_end]
    n = nf = npaid = 0; dp, da, fs, tot = [], [], [], []
    for i0 in starts:
        st, tau = run_eval_firm(spec.eval_phases, x, lo, nd, i0); n += 1
        if st != 1:
            continue
        nf += 1; act = min(i0 + tau + 1, n_end - 1); dp.append((d[act] - d[i0]).days)
        tm = []; pays, *_ = run_funded_firm(spec.funded, x, lo, nd, act, plan, atr, tm)
        tot.append(sum(pays))
        if pays:
            npaid += 1; da.append((d[min(tm[0], n_end - 1)] - d[act]).days); fs.append(pays[0])
    f = lambda v: float(np.median(v)) if v else np.nan
    return dict(pass_rate=nf / max(n, 1), days_to_pass=f(dp), p_paid=npaid / max(nf, 1), days_activation_to_payout=f(da), first_payout=float(np.mean(fs)) if fs else np.nan,
                paid_per_funded=float(np.mean(tot)) if tot else np.nan)


def main() -> None:
    s = build_stream(); d = pd.DatetimeIndex(s.dates)
    n_dev = int(np.searchsorted(s.dates, DEV_END, side="right"))
    idx = lambda dt: int(np.searchsorted(s.dates, np.datetime64(dt)))
    S = specs(); rows, best = [], {}
    for name, meta in S.items():
        grid = []
        for R, a1, a2, t1 in itertools.product(RS, A1S, A2S, meta["t1_grid"]):
            pl = mkplan(meta, t1, a1, a2)
            nets = np.array([farm(s, meta, R, pl, idx(f"{y}-01-01"), min(idx(f"{y}-01-01") + 756, n_dev))["net"] for y in YEARS])
            grid.append(dict(firm=name, R=R, a1=a1, a2=a2, t1=t1, mean=nets.mean(), p25=np.percentile(nets, 25), worst=nets.min(), green=int((nets > 0).sum()), nets=[round(v) for v in nets]))
        g = pd.DataFrame(grid); rows.append(g)
        feas = g[g["mean"] > 0].sort_values(["p25", "mean"], ascending=False)
        b = (feas if len(feas) else g.sort_values(["p25", "mean"], ascending=False)).iloc[0]
        best[name] = (b, len(feas) > 0)
    G = pd.concat(rows, ignore_index=True); G.drop(columns="nets").to_csv("results/firm_dev_grid.csv", index=False)
    first = int(np.flatnonzero(s.ok)[0]) + 260; starts = list(range(first, n_dev - 300, 3))
    starts_hold = list(range(idx("2023-01-01"), len(d) - 300, 3))
    mu = float(np.mean(s.pnl[(s.dates <= DEV_END) & s.ok] / s.atr_usd[(s.dates <= DEV_END) & s.ok]))
    dm = Stream(s.name, s.dates, np.where(s.ok, s.pnl - mu * np.nan_to_num(s.atr_usd), s.pnl), np.minimum(s.low - (s.pnl - np.where(s.ok, s.pnl - mu * np.nan_to_num(s.atr_usd), s.pnl)), 0.0), s.atr_usd, s.ok)
    out = []
    for name, (b, viable) in best.items():
        meta = S[name]; pl = mkplan(meta, b.t1, b.a1, b.a2)
        tm_dev = timing(s, meta, b.R, pl, n_dev, starts); tm_hold = timing(s, meta, b.R, pl, len(d), starts_hold)
        full = farm(s, meta, b.R, pl, int(np.flatnonzero(s.ok)[0]))
        seen = [round(farm(s, meta, b.R, pl, idx(x))["net"]) for x in ("2021-01-04", "2023-01-02", "2025-01-02")]
        plc = np.mean([farm(dm, meta, b.R, pl, idx(f"{y}-01-01"), min(idx(f"{y}-01-01") + 756, n_dev))["net"] for y in YEARS])
        fees_dev = np.mean([farm(s, meta, b.R, pl, idx(f"{y}-01-01"), min(idx(f"{y}-01-01") + 756, n_dev))["fees"] for y in YEARS])
        out.append(dict(firm=name, viable=viable, R=b.R, a1=b.a1, a2=b.a2, t1=b.t1, dev_mean=b["mean"], dev_p25=b.p25, dev_worst=b.worst, green=b.green, dev_fees=fees_dev,
                        net_per_fee=b["mean"] / fees_dev if fees_dev else np.nan, pass_rate=tm_dev["pass_rate"], days_to_pass=tm_dev["days_to_pass"], p_paid=tm_dev["p_paid"],
                        days_act_to_payout=tm_dev["days_activation_to_payout"], first_payout=tm_dev["first_payout"], paid_per_funded=tm_dev["paid_per_funded"],
                        hold_pass=tm_hold["pass_rate"], hold_days_to_pass=tm_hold["days_to_pass"], hold_days_act_to_payout=tm_hold["days_activation_to_payout"], hold_p_paid=tm_hold["p_paid"],
                        full_net=full["net"], full_payouts=full["payouts"], full_fees=full["fees"], seen_2021_2023_2025=str(seen), placebo_dev_mean=plc))
    R = pd.DataFrame(out); R.to_csv("results/firm_comparison.csv", index=False); pickle.dump({k: v[0].to_dict() for k, v in best.items()}, open("results/firm_best_configs.pkl", "wb"))
    print(R[["firm", "viable", "R", "a1", "a2", "t1", "dev_mean", "dev_p25", "dev_worst", "green", "net_per_fee", "placebo_dev_mean"]].round(2).to_string(index=False))
    print("\nTIME (dev starts): pass rate | calendar days start->pass | P(paid|funded) | days activation->first payout | first payout $ | paid per funded $")
    print(R[["firm", "pass_rate", "days_to_pass", "p_paid", "days_act_to_payout", "first_payout", "paid_per_funded"]].round(2).to_string(index=False))
    print("\nTIME (2023+ starts, seen):"); print(R[["firm", "hold_pass", "hold_days_to_pass", "hold_p_paid", "hold_days_act_to_payout"]].round(2).to_string(index=False))
    print("\nFULL history / seen starts:"); print(R[["firm", "full_net", "full_payouts", "full_fees", "seen_2021_2023_2025"]].round(0).to_string(index=False))


if __name__ == "__main__":
    main()
