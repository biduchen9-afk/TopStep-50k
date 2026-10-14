"""Pre-declared study (stage 1 = DEV only, 2010-2022):
  streams   : 3 sleeve sets x 4 switching rules                      (12)
  risk unit : ATR | trailing-40d strategy P&L std                     (2)
  Combine R : 3200 | 4800 | 6400  ($ risk per ATR-unit, fixed)        (3)
  funded    : (a1, a2) in {.2,.4} x {.4,.8,1.6}                        (6)
  farm      : refill {all, hyst, buffer} x pause_below {0, .10, .15}   (9)
Objective on 12 three-year DEV windows: 0.5 * (mean net + 10th-percentile net). Hold-out is NOT used to choose anything.
Writes results/regime_dev_grid.pkl + prints the top configs."""
from __future__ import annotations
import itertools, pickle, sys
import numpy as np, pandas as pd
import intraday_strategies as st
from contract_eval import CombineSizing
from farm_real import FarmSpec, run_farm
from portfolio import DEV_END, basket, dev_weights, make_sleeve
from risk_units import with_risk_unit
from run_corrected import NEW, cfg, rules
from run_farm_dev import plan
from run_dynamic_contract import PATH
from switching import switched_basket

pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 70)


def sleeve_set():
    C = st.CASH_OPEN
    S = dict(
        nq_noise=make_sleeve("nq", "noise_area", {}, costs=NEW, name="nq_noise"),
        nq_vwap=make_sleeve("nq", "vwap_atr", dict(k=0.3, s=0.2, start_min=C), costs=NEW, name="nq_vwap"),
        nq_gap30=make_sleeve("nq", "vv_filt", dict(filt="gap30", k=0.3, s=0.2), costs=NEW, name="nq_gap30"),
        es_noise=make_sleeve("es", "noise_area", {}, costs=NEW, name="es_noise"),
        es_vwap_t250=make_sleeve("es", "vwap_atr", dict(k=0.3, s=0.2, start_min=C), dict(trend=250), costs=NEW, name="es_vwap_t250"),
        gc_noise_t60=make_sleeve("gc", "noise_area", {}, dict(trend=60), costs=NEW, name="gc_noise_t60"))
    return S


SETS = {"P0 current 5": ["nq_noise", "nq_vwap", "es_noise", "es_vwap_t250", "gc_noise_t60"],
        "P1 current + NQ gap30": ["nq_noise", "nq_vwap", "nq_gap30", "es_noise", "es_vwap_t250", "gc_noise_t60"],
        "P2 NQ only (noise, vwap, gap30)": ["nq_noise", "nq_vwap", "nq_gap30"]}
RULES_ = ("static", "gate120", "gate250", "tilt")


def build_streams(S=None):
    S = S or sleeve_set()
    out = {}
    for sname, names in SETS.items():
        sl = [S[n] for n in names]
        w = dev_weights(sl, shrink=0.5)
        for rule in RULES_:
            out[(sname, rule)] = switched_basket(sl, rule, w, f"{sname}|{rule}")
    return out, S


def windows(s, n_win=12):
    n_dev = int(np.searchsorted(s.dates, DEV_END, side="right"))
    first = int(np.flatnonzero(s.ok)[0]) + 260
    return np.linspace(first, n_dev - 770, n_win).astype(int)


def evaluate(s, o, sz, pl, spec):
    return run_farm([s], sz, None, pl, cfg, rules, PATH, int(o), spec, n_end=int(o) + 756)


def main() -> None:
    streams, S = build_streams()
    pickle.dump(streams, open("/home/user/data_raw/regime_streams.pkl", "wb"))
    rows = []
    farm_specs = [(rf, pb) for rf in ("all", "hyst", "buffer") for pb in (0.0, 0.10, 0.15)]
    for (sname, rule), base in streams.items():
        for ru, rw in (("atr", 40), ("pnlstd", 40)):
            s = with_risk_unit(base, ru, rw)
            wins = windows(s)
            for R in (3200, 4800, 6400):
                sz = CombineSizing("fixed", R)
                for a1, a2 in itertools.product((0.2, 0.4), (0.4, 0.8, 1.6)):
                    pl = plan(a1, a2)
                    for rf, pb in farm_specs:
                        spec = FarmSpec(max_evals=10, max_funded=5, p_est=1.0, refill=rf, pause_below=pb)
                        net = np.array([evaluate(s, o, sz, pl, spec)["net"] for o in wins])
                        rows.append(dict(set=sname, rule=rule, unit=ru, R=R, a1=a1, a2=a2, refill=rf, pause=pb, mean=net.mean(), p10=np.percentile(net, 10),
                                         worst=net.min(), obj=0.5 * (net.mean() + np.percentile(net, 10)), pos=float((net > 0).mean())))
        print(sname, rule, "done", flush=True)
    g = pd.DataFrame(rows); g.to_pickle("results/regime_dev_grid.pkl"); g.to_csv("results/regime_dev_grid.csv", index=False)
    print(f"{len(g)} configurations evaluated on 12 DEV windows")
    print(g.sort_values("obj", ascending=False).head(15).round(0).to_string(index=False))
    for col in ("set", "rule", "unit", "R", "a1", "a2", "refill", "pause"):
        print(f"\nmean dev objective by {col}:"); print(g.groupby(col).obj.mean().round(0).to_string())


if __name__ == "__main__":
    main()
