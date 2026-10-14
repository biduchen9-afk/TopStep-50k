"""Per-start evaluation of a Combine+funded configuration with fixed or cushion-proportional sizing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from pricing_config import EngineConfig
from real_engine import FundedPlan, RuleSet, Stream, combine_outcomes, run_funded_detail, year_bootstrap, _lists

PATH = "no_activation_dll"


@dataclass(frozen=True)
class CombineSizing:
    """fixed: R dollars per ATR move (integer baskets/micros). dyn: risk = a x cushion above the MLL."""
    kind: str = "fixed"
    value: float = 3200.0

    def label(self) -> str:
        return f"Rc{int(self.value)}" if self.kind == "fixed" else f"Cdyn{self.value}"


def contract_metrics(comb: Stream, fund: Stream, sizing: CombineSizing, plan: FundedPlan, Rf: Optional[float], cfg: EngineConfig,
                     rules: RuleSet, starts: np.ndarray, groups: np.ndarray, path: str = PATH) -> dict:
    xc, lc = _lists(comb)
    xf, lf = _lists(fund)
    n = len(xc)
    atr_c, atr_f = comb.atr_usd.tolist(), fund.atr_usd.tolist()
    if sizing.kind == "fixed":
        c = combine_outcomes(xc, lc, comb.desired_micros(sizing.value), cfg, path, rules, starts)
    else:
        c = combine_outcomes(xc, lc, [1] * n, cfg, path, rules, starts, dyn_a=sizing.value, atr=atr_c)
    ps = c["status"] == 1
    ndf = fund.desired_micros(Rf) if (Rf is not None and plan.a1 is None) else [1] * n
    tot = np.zeros(len(starts)); first = np.zeros(len(starts)); npay = np.zeros(len(starts)); days = np.zeros(len(starts))
    nopay_dead = np.zeros(len(starts), dtype=bool)
    for k in np.flatnonzero(ps):
        i_f = int(starts[k] + c["tau"][k])
        if i_f >= n - 5:
            continue
        pays, d, state, _ = run_funded_detail(xf, lf, ndf, i_f, rules, plan, atr_f)
        tot[k] = sum(pays); first[k] = pays[0] if pays else 0.0; npay[k] = len(pays); days[k] = d
        nopay_dead[k] = (not pays) and state == 1
    net = tot - c["fees"]
    paid = (first > 0) & ps
    nf = max(int(ps.sum()), 1)
    cost = c["fees"].sum() / nf
    ev = year_bootstrap(net, groups, np.mean, n_boot=300)
    return dict(p_pass=float(ps.mean()), tries=float(1 / ps.mean()) if ps.mean() else np.inf, e_tau=float(c["tau"].mean()),
                cost_funded=cost, p_first=paid.sum() / nf, first_net=float(first[paid].mean()) if paid.any() else 0.0,
                covers=bool(paid.any() and first[paid].mean() >= cost), payouts_per_funded=float(npay[ps].mean()) if ps.any() else 0.0,
                extraction_per_funded=float(tot[ps].mean()) if ps.any() else 0.0, p_die_no_payout=float(nopay_dead[ps].mean()) if ps.any() else 0.0,
                days_alive=float(days[ps].mean()) if ps.any() else 0.0, ev_funded=ev[0] / max(ps.mean(), 1e-9), ev_att=ev[0], ev_lo=ev[1], ev_hi=ev[2],
                n_funded=int(ps.sum()))
