"""Payouts cluster in the same months because every account trades the same stream on the same path. Does assigning accounts to DIFFERENT
streams (NQ-trend sleeves / NQ gap-VWAP / ES+GC sleeves) smooth the payout calendar? Same dev-window protocol; coverage figures are descriptive."""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
from contract_eval import CombineSizing
from farm_real import FarmSpec, run_farm
from portfolio import basket, dev_weights
from risk_units import with_risk_unit
from run_corrected import cfg, rules
from run_dynamic_contract import PATH
from run_monthly_payout import coverage, fplan, windows, idx_first
from run_regime_study import sleeve_set

S = sleeve_set()
allsl = list(S.values())
GROUPS = {"NQ trend": ["nq_noise", "nq_vwap"], "NQ gap-VWAP": ["nq_gap30"], "ES+GC": ["es_noise", "es_vwap_t250", "gc_noise_t60"]}


def group_stream(names):
    w = dev_weights([S[n] for n in names], shrink=0.5)
    ww = {s.name: w.get(s.name, 0.0) for s in allsl}
    return with_risk_unit(basket(allsl, weights=ww, name="+".join(names)), "atr", 40)


def main() -> None:
    st = {k: group_stream(v) for k, v in GROUPS.items()}
    for k, s in st.items():
        assert np.array_equal(s.dates, st["NQ trend"].dates), k
    combos = {"single: NQ trend": ["NQ trend"], "single: NQ gap-VWAP": ["NQ gap-VWAP"], "single: ES+GC": ["ES+GC"],
              "2 streams: NQ trend + NQ gap": ["NQ trend", "NQ gap-VWAP"], "2 streams: NQ trend + ES+GC": ["NQ trend", "ES+GC"],
              "3 streams: all": ["NQ trend", "NQ gap-VWAP", "ES+GC"]}
    SZ = CombineSizing("fixed", 9600.0)
    pl = fplan("standard", 3000, 0.4, 0.4)
    rows = []
    for lab, keys in combos.items():
        ss = [st[k] for k in keys]
        spec = FarmSpec(max_evals=10, max_funded=5, p_est=1.0, refill="hyst")
        wins = windows(ss[0])
        net = np.array([run_farm(ss, SZ, None, pl, cfg, rules, PATH, int(o), spec, n_end=int(o) + 756)["net"] for o in wins])
        full = run_farm(ss, SZ, None, pl, cfg, rules, PATH, idx_first(ss[0]), spec, log=True)
        rows.append(dict(streams=lab, dev_mean=net.mean(), dev_p10=np.percentile(net, 10), dev_pos=float((net > 0).mean()), payouts=full["n_payouts"], **coverage(full, ss[0])))
    g = pd.DataFrame(rows); g.to_csv("results/monthly_payout_diversified.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(g.round(2).to_string(index=False))


if __name__ == "__main__":
    sys.argv = sys.argv[:1]
    main()
