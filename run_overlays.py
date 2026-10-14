"""Trade-management overlays tested on the existing NQ basket (not counted as trials; pre-registration 8): tighter soft daily loss limit, daily profit cap, trade-count limiter.
Metric: ten 3-year dev farms (2011..2020 starts), balanced default, plus seen starts."""
from __future__ import annotations
import sys
from dataclasses import replace
import numpy as np, pandas as pd
import intraday_strategies as st
from final_strategy import FINAL
from firm_specs import specs
from multi_firm import FirmPlan, Policy, run_multifirm
from portfolio import DEV_END, basket, dev_weights, make_sleeve
from real_engine import FundedPlan
from risk_units import with_risk_unit
from run_corrected import NEW

pd.set_option("display.width", 250, "display.max_columns", 40)
C = st.CASH_OPEN
TS = specs()["Topstep 150K"]["spec"]


def stream(max_trades=3):
    mt = {} if max_trades == 3 else dict(max_trades=max_trades)
    sl = [make_sleeve("nq", "noise_area", {}, mt or None, costs=NEW, name="nq_noise"), make_sleeve("nq", "vwap_atr", dict(k=0.3, s=0.2, start_min=C, **mt), costs=NEW, name="nq_vwap"),
          make_sleeve("nq", "vv_filt", dict(filt="gap30", k=0.3, s=0.2, **mt), costs=NEW, name="nq_gap30")]
    return basket(sl, weights=dev_weights(sl, shrink=0.5), name=f"nq3_k{max_trades}")


def spec_with(dll=None, cap_eval=None, cap_funded=None):
    ph = replace(TS.eval_phases[0], dll=TS.eval_phases[0].dll if dll is None else dll, day_cap=cap_eval)
    fd = replace(TS.funded, dll=TS.funded.dll if dll is None else dll, day_cap=cap_funded)
    return replace(TS, eval_phases=(ph,), funded=fd, max_evals=10)


def evaluate(s, spec, label):
    d = pd.DatetimeIndex(s.dates); n_dev = int(np.searchsorted(s.dates, DEV_END, side="right"))
    idx = lambda dt: int(np.searchsorted(s.dates, np.datetime64(dt)))
    pl = FundedPlan("p", "x", FINAL["t1"], 0.5, 1.0, 0.0, 0.5, 1.0, 99, FINAL["a1"], FINAL["a2"])
    fp = FirmPlan(spec, [s], 9600.0, pl, p_est=1.0, refill="hyst")
    nets = np.array([run_multifirm([fp], Policy(reserve_cycles=None), idx(f"{y}-01-01"), min(idx(f"{y}-01-01") + 756, n_dev))["net"] for y in range(2011, 2021)])
    seen = [round(run_multifirm([fp], Policy(reserve_cycles=None), idx(x))["net"]) for x in ("2021-01-04", "2023-01-02", "2025-01-02", "2025-07-01")]
    full = run_multifirm([fp], Policy(reserve_cycles=None), int(np.flatnonzero(s.ok)[0]))
    return dict(case=label, dev_mean=nets.mean(), dev_p25=np.percentile(nets, 25), dev_worst=nets.min(), green=int((nets > 0).sum()), seen=str(seen), full_net=full["net"], full_fees=full["fees"])


def main() -> None:
    base = with_risk_unit(stream(3), "atr", 40); rows = [evaluate(base, spec_with(), "baseline (balanced, soft DLL -3,000, up to 3 trades)")]
    for dll in (-2400.0, -2000.0, -1500.0):
        rows.append(evaluate(base, spec_with(dll=dll), f"daily-loss breaker: soft limit {dll:,.0f}"))
    for cap in (3600.0, 2700.0, 1800.0):
        rows.append(evaluate(base, spec_with(cap_eval=cap), f"daily profit cap {cap:,.0f} in the Combine only"))
    rows.append(evaluate(base, spec_with(cap_eval=2700.0, cap_funded=2700.0), "daily profit cap 2,700 in Combine and funded"))
    for k in (1, 2):
        rows.append(evaluate(with_risk_unit(stream(k), "atr", 40), spec_with(), f"trade-count limiter: max {k} entries per sleeve per day"))
    R = pd.DataFrame(rows); R.to_csv("results/overlays.csv", index=False)
    print(R.round(0).to_string(index=False))


if __name__ == "__main__":
    main()
