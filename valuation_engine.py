"""Valuation of Topstep Combine -> XFA cycles: p, E[tau], mu_attempt, Cost_funded, EV.

Unit convention (never mixed):
  * per-attempt:       p, mu_attempt, EV_attempt
  * per-funded-cycle:  Cost_funded = mu_attempt / p + C_act, EV_cycle = EV_attempt / p
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from combine_simulator import AttemptResults, simulate_attempts, simulate_xfa
from pricing_config import EngineConfig, FeePath, load_config

BILLING_CYCLE_DAYS = 21
_DEFAULT_CFG: Optional[EngineConfig] = None


def _default_cfg() -> EngineConfig:
    global _DEFAULT_CFG
    if _DEFAULT_CFG is None:
        _DEFAULT_CFG = load_config()
    return _DEFAULT_CFG


# ----------------------------------------------------------------------------
# Closed-form cash-flow / EV identities
# ----------------------------------------------------------------------------
def n_bills(days_elapsed: int, cycle: int = BILLING_CYCLE_DAYS) -> int:
    """N_bills = 1 + floor((tau - 1) / 21): Day-0 bill plus each completed renewal."""
    return 1 + (max(int(days_elapsed), 1) - 1) // cycle


def calc_attempt_cashflow(path_key: str, days_elapsed: int, passed: bool,
                          cfg: Optional[EngineConfig] = None) -> float:
    """Fees paid for one fresh attempt; activation is charged only on PASSED."""
    cfg = cfg or _default_cfg()
    path = cfg.paths[path_key]
    fees = n_bills(days_elapsed, cfg.billing_cycle_days) * path.monthly_cost
    return fees + (path.activation_fee if passed else 0.0)


def calc_cost_per_funded(p: float, mu_attempt: float, c_act: float) -> float:
    return mu_attempt / p + c_act


def calc_ev_per_attempt(p: float, mu_ext: float, mu_attempt: float, c_act: float) -> float:
    return p * (mu_ext - c_act) - mu_attempt


def calc_ev_per_cycle(p: float, mu_ext: float, mu_attempt: float, c_act: float) -> float:
    return calc_ev_per_attempt(p, mu_ext, mu_attempt, c_act) / p


def calc_breakeven_extraction(p: float, mu_attempt: float, c_act: float) -> float:
    return calc_cost_per_funded(p, mu_attempt, c_act)


def analytic_pass_probability(mu: float, sigma: float, target: float, drawdown: float) -> float:
    """P(reach +target before a trailing drawdown of `drawdown`) for continuous BM.

    The running max at the first drawdown of size b is Exponential with mean
    m = sigma^2 / (2 mu) * (exp(2 mu b / sigma^2) - 1)  (Lehoczky 1977; m = b when mu = 0),
    so p = exp(-target / m). Ignores DLL, consistency rule and EOD discreteness.
    """
    k = 2.0 * mu / sigma ** 2
    m = drawdown if abs(k * drawdown) < 1e-12 else (math.exp(k * drawdown) - 1.0) / k
    return math.exp(-target / m)


# ----------------------------------------------------------------------------
# Breach-policy sequencing (billing clock carried across attempts)
# ----------------------------------------------------------------------------
@dataclass
class ChainStats:
    attempts: int
    funded: int
    fees: float          # subscription + reset fees, excluding activation
    days: float          # trading days consumed, including idle days


def run_attempt_chain(res: AttemptResults, path: FeePath, policy: str,
                      cycle: int = BILLING_CYCLE_DAYS) -> ChainStats:
    """Replay i.i.d. attempts in order under a breach policy until the pool is used up.

    `c` = days already consumed in the current paid billing window (0..cycle).
    An attempt of length tau needs ceil((c + tau) / cycle) windows in total, one of
    which is already paid, so it triggers ceil((c + tau) / cycle) - 1 rebills.
    """
    attempts = funded = 0
    fees = days = 0.0
    chain_fees = chain_days = 0.0
    chain_attempts = 0
    c = None  # None => next attempt starts with a fresh C_sub bill
    for tau, ok in zip(res.tau.tolist(), res.passed.tolist()):
        if c is None:
            chain_fees += path.monthly_cost
            c = 0
        windows = -(-(c + tau) // cycle)
        chain_fees += (windows - 1) * path.monthly_cost
        used = c + tau - (windows - 1) * cycle   # in (0, cycle]
        chain_days += tau
        chain_attempts += 1
        if ok:
            attempts += chain_attempts
            funded += 1
            fees += chain_fees
            days += chain_days
            chain_fees = chain_days = 0.0
            chain_attempts = 0
            c = None
            continue
        if policy == "immediate_reset" and path.reset_fee < path.monthly_cost and used < cycle:
            chain_fees += path.reset_fee
            c = used
        elif policy == "wait_for_rebill":
            chain_days += cycle - used   # idle until the renewal grants a reset
            c = None
        else:  # cancel_and_reopen, or immediate_reset when reopening is cheaper
            c = None
    return ChainStats(attempts=attempts, funded=funded, fees=fees, days=days)


# ----------------------------------------------------------------------------
# Full valuation
# ----------------------------------------------------------------------------
@dataclass
class PathValuation:
    path: str
    sharpe_gross: float
    sharpe_net: float
    p_empirical: float
    p_analytic: float
    e_tau: float
    mu_attempt: float
    cost_funded: float
    mu_ext: float
    p_xfa_payout: float
    ev_attempt: float
    ev_cycle: float
    mu_ext_star: float
    days_to_funded: float
    truncated: int


def value_path(cfg: EngineConfig, path_key: str, seed: Optional[int] = None) -> PathValuation:
    path = cfg.paths[path_key]
    n = cfg.farm_policy.n_simulations
    rng = np.random.default_rng(cfg.farm_policy.seed if seed is None else seed)

    res = simulate_attempts(cfg, path, n, rng)
    chain = run_attempt_chain(res, path, cfg.farm_policy.breach_action, cfg.billing_cycle_days)
    if chain.funded == 0:
        p = 0.0
        mu_attempt = float(np.mean([calc_attempt_cashflow(path_key, t, False, cfg) for t in res.tau]))
    else:
        p = chain.funded / chain.attempts
        mu_attempt = chain.fees / chain.attempts

    gross = simulate_xfa(cfg, path, n, rng)
    mu_ext = float(np.mean(gross)) * cfg.profit_split

    mp = cfg.market_process
    mu_net = mp.mu_daily_gross - cfg.friction.daily_cost
    sharpe_net = mu_net / mp.sigma_daily * math.sqrt(252)
    p_an = analytic_pass_probability(mu_net, mp.sigma_daily,
                                     cfg.combine_rules.profit_target, cfg.combine_rules.trailing_mll)

    c_act = path.activation_fee
    if p > 0:
        cost_funded = calc_cost_per_funded(p, mu_attempt, c_act)
        ev_cycle = calc_ev_per_cycle(p, mu_ext, mu_attempt, c_act)
        days_to_funded = chain.days / chain.funded
    else:
        cost_funded = ev_cycle = days_to_funded = float("inf")
        ev_cycle = float("-inf")
    return PathValuation(
        path=path_key,
        sharpe_gross=mp.effective_sharpe,
        sharpe_net=sharpe_net,
        p_empirical=p,
        p_analytic=p_an,
        e_tau=float(res.tau.mean()),
        mu_attempt=mu_attempt,
        cost_funded=cost_funded,
        mu_ext=mu_ext,
        p_xfa_payout=float(np.mean(gross > 0)),
        ev_attempt=calc_ev_per_attempt(p, mu_ext, mu_attempt, c_act),
        ev_cycle=ev_cycle,
        mu_ext_star=cost_funded,
        days_to_funded=days_to_funded,
        truncated=int(res.truncated.sum()),
    )


def breakeven_sharpe(cfg: EngineConfig, path_key: str, lo: float = 0.0, hi: float = 3.0,
                     iters: int = 12) -> float:
    """Smallest gross annual Sharpe with EV_attempt >= 0 (bisection, common random numbers)."""
    def ev(sr: float) -> float:
        c = cfg.with_market(eval_mode="positive_drift", sharpe_annual=sr)
        return value_path(c, path_key).ev_attempt

    if ev(lo) >= 0:
        return lo
    if ev(hi) < 0:
        return float("nan")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if ev(mid) >= 0 else (mid, hi)
    return hi
