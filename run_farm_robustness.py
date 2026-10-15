"""Hold-out sensitivity of the frozen farm (a1=0.4, <=10-20 evals, p_est=1.0): post-payout aggressiveness, slippage, fee path, basket-low assumption."""
from __future__ import annotations

import pickle
import numpy as np
import pandas as pd

import intraday_strategies as st
from pricing_config import load_config
from real_engine import FundedPlan, rules_from_config
from contract_eval import CombineSizing
from farm_real import FarmSpec, run_farm
from portfolio import DEV_END, basket, make_sleeve
from run_farm_dev import plan, SZ
from run_dynamic_contract import PATH

pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 70)
cfg = load_config()
W = pickle.load(open("/home/user/data_raw/portfolio_weights.pkl", "rb"))


def build(costs=None):
    S = [make_sleeve("nq", "noise_area", {}, costs=costs, name="nq_noise"), make_sleeve("nq", "vwap_atr", dict(k=0.3, s=0.2, start_min=st.CASH_OPEN), costs=costs, name="nq_vwap"),
         make_sleeve("es", "noise_area", {}, costs=costs, name="es_noise"), make_sleeve("es", "vwap_atr", dict(k=0.3, s=0.2, start_min=st.CASH_OPEN), dict(trend=250), costs=costs, name="es_vwap_t250"),
         make_sleeve("gc", "noise_area", {}, dict(trend=60), costs=costs, name="gc_noise_t60")]
    return {"NQ noise-area": basket([S[0]], name="nq"), "portfolio": basket(S, weights=W, name="pf"), "NQ-only sleeves (noise+vwap)": basket(S[:2], weights={k: W[k] for k in ("nq_noise", "nq_vwap")}, name="nq2")}


def farm(stream, pl, path=PATH, evals=10, **ov):
    n_h = int(np.searchsorted(stream.dates, DEV_END, side="right"))
    yrs = (stream.dates[-1] - stream.dates[n_h]).astype(float) / 365.25
    rules = rules_from_config(cfg, path, **ov)
    r = pd.DataFrame([run_farm([stream], SZ, None, pl, cfg, rules, path, o, FarmSpec(max_evals=evals, max_funded=5, p_est=1.0)) for o in range(n_h, n_h + 60, 10)])
    return dict(net=r.net.mean(), per_year=r.net.mean() / yrs, worst=r.net.min(), funded=r.n_funded.mean(), payouts=r.payouts.mean(), fees=r.fees.mean(), avg_funded=r.avg_funded_active.mean())


def main() -> None:
    base = build()
    rows = []
    for name, s in base.items():
        for a2 in (0.1, 0.2, 0.4, 0.8, 1.6):
            rows.append(dict(test="post-payout aggressiveness a2", stream=name, setting=f"a1=0.4 a2={a2}", **farm(s, plan(0.4, a2))))
    for slip in (2.0, 3.0):
        b = build(st.Costs(slippage_ticks_rt=slip))
        for name in ("NQ noise-area", "portfolio"):
            for a2 in (0.4, 1.6):
                rows.append(dict(test=f"slippage {slip:.0f} ticks", stream=name, setting=f"a2={a2}", **farm(b[name], plan(0.4, a2))))
    for pth in ("standard", "no_activation"):
        for name in ("NQ noise-area", "portfolio"):
            for a2 in (0.4, 1.6):
                rows.append(dict(test=f"fee path {pth} (no DLL)", stream=name, setting=f"a2={a2}", **farm(base[name], plan(0.4, a2), path=pth)))
    for name in ("NQ noise-area", "portfolio"):
        for lab, ov in (("breach checked only at the close", dict(intraday_mll=False)), ("10 winning days needed", dict(standard_win_days=10)), ("payout cap $2,500", dict(cap_standard=2500.0))):
            rows.append(dict(test=lab, stream=name, setting="a1=0.4 a2=0.4", **farm(base[name], plan(0.4, 0.4), **ov)))
    df = pd.DataFrame(rows); df.to_csv("results/farm_robustness.csv", index=False)
    print(df.round(0).to_string(index=False))


if __name__ == "__main__":
    main()
