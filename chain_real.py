"""Sequential rinse-and-repeat on real data with the spec's three breach policies and billing clock.

Billing (spec section 2.3): the Combine bills C_sub on its first session and every 21 sessions while
active. Policies after an EOD/real-time MLL breach:
  cancel_and_reopen    cancel and open a fresh subscription (new clock, pays C_sub).
  immediate_reset      pay C_reset now and KEEP the old billing clock; if C_reset >= C_sub the spec says to
                       cancel and reopen instead ("whichever is cheaper"), which is what happens here.
  immediate_reset_keep forced version that always pays C_reset and keeps the old clock (shows its extra cost).
  wait_for_rebill      sit idle until the next 21-session renewal, which grants a complimentary reset.
Activation fee is paid only on a pass; the Combine subscription then auto-cancels.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from pricing_config import EngineConfig
from real_engine import FundedPlan, RuleSet, Stream, run_combine, run_funded_detail, _lists

POLICIES = ("cancel_and_reopen", "immediate_reset", "immediate_reset_keep", "wait_for_rebill")


def combine_chain(xc, lc, ndc, t0: int, n_end: int, rules: RuleSet, cfg: EngineConfig, path_key: str,
                  policy: str) -> Tuple[Optional[int], float, int, int]:
    """Run attempts back to back from session t0 until one passes or the data ends.

    Returns (pass_session or None, fees, attempts, last_session_used).
    """
    if policy not in POLICIES:
        raise ValueError(policy)
    p = cfg.paths[path_key]
    cyc = cfg.billing_cycle_days
    sub, rst, act = p.monthly_cost, p.reset_fee, p.activation_fee
    keep_clock = policy == "immediate_reset_keep" or (policy == "immediate_reset" and rst < sub)
    wait = policy == "wait_for_rebill"
    t, c0, fees, attempts = t0, t0, 0.0, 0
    first = True
    while t < n_end - 5:
        if first:
            fees += sub
            c0 = t
            first = False
        elif wait:
            pass                                    # the renewal bill was added when the idle period ended
        elif keep_clock:
            fees += rst
            if t > c0 and (t - c0) % cyc == 0:
                fees += sub                         # renewal falls on the reset day
        else:
            fees += sub
            c0 = t
        status, tau = run_combine(xc, lc, ndc, t, rules)
        end = t + tau                               # first session after the attempt
        if keep_clock or wait:
            lo_s = t + 1
            fees += sub * sum(1 for s in range(lo_s, end) if s > c0 and (s - c0) % cyc == 0)
        else:
            fees += sub * ((max(tau, 1) - 1) // cyc)
        attempts += 1
        if status == 1:
            return end - 1, fees + act, attempts, end - 1
        if wait:
            s = c0 + cyc * math.ceil((end - c0) / cyc)
            fees += sub                             # renewal that grants the free reset
            t = s
        else:
            t = end
    return None, fees, attempts, min(t, n_end - 1)


def policy_stats(comb: Stream, Rc: float, cfg: EngineConfig, path_key: str, rules: RuleSet,
                 starts: np.ndarray, policy: str) -> Dict[str, float]:
    xc, lc = _lists(comb)
    nd = comb.desired_micros(Rc)
    n = len(xc)
    fees, days, att = [], [], []
    for i0 in starts:
        sp, f, a, _ = combine_chain(xc, lc, nd, int(i0), n, rules, cfg, path_key, policy)
        if sp is not None:
            fees.append(f); days.append(sp + 1 - int(i0)); att.append(a)
    return dict(policy=policy, funded_chains=len(fees), fees_to_funded=float(np.mean(fees)),
                days_to_funded=float(np.mean(days)), attempts_to_funded=float(np.mean(att)))


def replay_detail(comb: Stream, Rc: float, fund: Stream, Rf: float, plan: FundedPlan, cfg: EngineConfig,
                  rules: RuleSet, path_key: str, i0: int, max_sessions: Optional[int] = None,
                  policy: str = "cancel_and_reopen") -> Dict[str, float]:
    """One account at a time: Combine -> funded -> payouts -> close/die -> new Combine, to the horizon."""
    xc, lc = _lists(comb)
    xf, lf = _lists(fund)
    n_end = len(xc) if max_sessions is None else min(len(xc), i0 + max_sessions)
    xc, lc, xf, lf = xc[:n_end], lc[:n_end], xf[:n_end], lf[:n_end]
    ndc, ndf = comb.desired_micros(Rc)[:n_end], fund.desired_micros(Rf)[:n_end]
    t = i0
    out = dict(fees=0.0, payouts=0.0, attempts=0, funded=0, n_payouts=0, days_combine=0, days_funded=0,
               left_on_table=0.0, died=0, closed=0)
    while t < n_end - 5:
        sp, fees, att, last = combine_chain(xc, lc, ndc, t, n_end, rules, cfg, path_key, policy)
        out["fees"] += fees
        out["attempts"] += att
        out["days_combine"] += (sp + 1 - t) if sp is not None else (n_end - t)
        if sp is None:
            break
        out["funded"] += 1
        i_f = sp + 1
        if i_f >= n_end - 5:
            break
        pays, days, state, bal = run_funded_detail(xf, lf, ndf, i_f, rules, plan)
        out["payouts"] += sum(pays)
        out["n_payouts"] += len(pays)
        out["days_funded"] += days
        if state == 2:
            out["closed"] += 1
            out["left_on_table"] += max(bal, 0.0)
        elif state == 1:
            out["died"] += 1
        t = i_f + days
        if state == 0:
            break
    out["net"] = out["payouts"] - out["fees"]
    out["sessions"] = n_end - i0
    return out
