from dataclasses import replace

import numpy as np
import pytest

from combine_simulator import AttemptResults, simulate_single_day
from pricing_config import load_config
from valuation_engine import (
    analytic_pass_probability,
    calc_attempt_cashflow,
    calc_cost_per_funded,
    calc_ev_per_attempt,
    calc_ev_per_cycle,
    run_attempt_chain,
    value_path,
)


def test_billing_and_activation_separation():
    """Activation fee is charged ONLY upon passing, never on active/breached months."""
    assert calc_attempt_cashflow("standard", days_elapsed=15, passed=False) == 199.0
    assert calc_attempt_cashflow("standard", days_elapsed=15, passed=True) == 348.0   # 199 + 149
    assert calc_attempt_cashflow("standard", days_elapsed=25, passed=False) == 398.0  # 2 billings (Day 0, Day 21)
    assert calc_attempt_cashflow("standard", days_elapsed=25, passed=True) == 547.0   # 2*199 + 149


def test_no_activation_path_cashflows():
    """Verify No Activation and DLL-discounted paths."""
    assert calc_attempt_cashflow("no_activation", days_elapsed=15, passed=True) == 229.0
    assert calc_attempt_cashflow("no_activation", days_elapsed=25, passed=True) == 458.0
    assert calc_attempt_cashflow("no_activation_dll", days_elapsed=15, passed=True) == 199.0


def test_ev_unit_consistency():
    """Ensure EV_cycle == EV_attempt / p and Breakeven == Cost_funded (no double-division by p)."""
    p = 0.25
    mu_attempt = 199.0
    c_act = 149.0

    cost_funded = calc_cost_per_funded(p=p, mu_attempt=mu_attempt, c_act=c_act)
    assert pytest.approx(cost_funded) == 945.0

    ev_attempt = calc_ev_per_attempt(p=p, mu_ext=945.0, mu_attempt=mu_attempt, c_act=c_act)
    ev_cycle = calc_ev_per_cycle(p=p, mu_ext=945.0, mu_attempt=mu_attempt, c_act=c_act)
    assert pytest.approx(ev_attempt) == 0.0
    assert pytest.approx(ev_cycle) == 0.0


def test_dll_truncates_left_tail():
    """Verify that enabling the -$3,000 DLL caps single-day losses at -$3,000."""
    daily_pnl = simulate_single_day(raw_pnl=-4200.0, intraday_dll=-3000.0)
    assert daily_pnl == -3000.0


# --- Additional invariants ---------------------------------------------------

def test_chain_billing_matches_closed_form():
    """Under cancel_and_reopen, chain fees equal the sum of per-attempt N_bills * C_sub."""
    cfg = load_config()
    path = cfg.paths["standard"]
    res = AttemptResults(tau=np.array([15, 25, 42, 43, 10]),
                         passed=np.array([False, False, False, False, True]),
                         truncated=np.zeros(5, dtype=bool))
    chain = run_attempt_chain(res, path, "cancel_and_reopen")
    expected = sum(calc_attempt_cashflow("standard", t, False, cfg) for t in res.tau)
    assert chain.funded == 1 and chain.attempts == 5
    assert chain.fees == pytest.approx(expected)


def test_wait_for_rebill_costs_time_not_money():
    cfg = load_config()
    path = cfg.paths["standard"]
    res = AttemptResults(tau=np.array([5, 10]), passed=np.array([False, True]),
                         truncated=np.zeros(2, dtype=bool))
    reopen = run_attempt_chain(res, path, "cancel_and_reopen")
    wait = run_attempt_chain(res, path, "wait_for_rebill")
    assert wait.fees == reopen.fees == 2 * 199.0
    assert wait.days == 21 + 10 and reopen.days == 5 + 10


def test_no_edge_mc_matches_analytic_trailing_drawdown():
    """With no edge, no friction, no DLL, p ~= exp(-9000/4500) (consistency rule rarely binds)."""
    cfg = load_config()
    cfg = cfg.with_market(eval_mode="no_edge", sigma_daily=150.0)  # fine steps -> near-continuous
    cfg = replace(cfg, friction=replace(cfg.friction, trades_per_day=0),
                  combine_rules=replace(cfg.combine_rules, max_days=20000))
    cfg = cfg.with_policy(n_simulations=20000)
    v = value_path(cfg, "standard")
    assert v.p_empirical == pytest.approx(analytic_pass_probability(0.0, 150.0, 9000, 4500), abs=0.02)


def test_cost_funded_identity_from_simulation():
    cfg = load_config().with_policy(n_simulations=4000)
    v = value_path(cfg, "standard")
    assert v.ev_cycle == pytest.approx(v.ev_attempt / v.p_empirical)
    assert v.mu_ext_star == pytest.approx(v.mu_attempt / v.p_empirical + 149.0)


def test_lifecycle_billing_clock_when_nothing_happens():
    """Tiny volatility, no friction: the account never resolves, so fees = ceil(756/21) bills."""
    from lifecycle_simulator import simulate_lifecycle
    cfg = load_config().with_market(eval_mode="no_edge", sigma_daily=1.0)
    cfg = replace(cfg, friction=replace(cfg.friction, trades_per_day=0))
    r = simulate_lifecycle(cfg, "standard", 50, np.random.default_rng(0))
    assert np.all(r.fees == 36 * 199.0)
    assert np.all(r.total_pnl == -36 * 199.0) and np.all(r.n_funded == 0)


# --- Real-data engine ----------------------------------------------------------------------

def test_session_open_follows_us_dst_in_phnom_penh():
    """18:00 ET is 05:00 ICT in US summer time and 06:00 ICT in winter: never hard-code a local hour."""
    from datetime import datetime
    from market_data import session_open_phnom_penh
    assert session_open_phnom_penh(datetime(2024, 7, 10)).strftime("%H:%M") == "05:00"
    assert session_open_phnom_penh(datetime(2024, 12, 10)).strftime("%H:%M") == "06:00"


def test_combine_consistency_and_breach():
    from real_engine import run_combine
    zeros = [0.0] * 3
    # a $5,000 day inside a $9,500 total breaks the 50% rule; one more green day then passes it
    assert run_combine([5000, 4500, 1000], zeros, 0, None, 9000, 4500, 0.5) == (1, 3)
    assert run_combine([5000, 4500], [0.0, 0.0], 0, None, 9000, 4500, 0.5) == (-1, 2)
    # trailing MLL: +3000 then -4500 from the peak breaches on day 2
    assert run_combine([3000, -4500], [0.0, 0.0], 0, None, 9000, 4500, 0.5) == (0, 2)


def test_combine_dll_liquidates_on_intraday_low():
    from real_engine import run_combine
    # the close is +100 but the intraday low touched -3,000: the day is booked at the DLL
    status, tau = run_combine([100.0] * 3, [-3200.0, 0.0, 0.0], 0, -3000.0, 9000, 4500, 0.5)
    assert (status, tau) == (-1, 3)


def test_funded_cushion_payout_and_death():
    from real_engine import FundedVariant, run_funded
    v = FundedVariant("t", t1=2000, w1=0.5, t2=None, need_win_days=5)
    # +300/day: balance 2,100 on day 7 with 7 winning days -> withdraw 50% = 1,050 gross, 945 net
    p1, p2, days, state = run_funded([300.0] * 20, [0.0] * 20, 0, None, v, 4500, 0.9)
    assert (round(p1, 6), p2, days, state) == (945.0, 0.0, 7, 2)
    # five -1,000 days: floor is -4,500 (peak 0), breached on day 5 -> dead, no payout
    assert run_funded([-1000.0] * 6, [0.0] * 6, 0, None, v, 4500, 0.9) == (0.0, 0.0, 5, 1)
    # the winning-day rule delays a payout even when the balance is already high
    big = FundedVariant("b", t1=2000, w1=0.5, t2=None, need_win_days=5)
    p1, _, days, _ = run_funded([2500.0, 0.0, 0.0, 0.0, 0.0, 200.0, 200.0, 200.0, 200.0], [0.0] * 9, 0, None, big, 4500, 0.9)
    assert days == 9 and p1 > 0   # waits for the 5th winning day (day 9), not day 1


def test_second_stage_uses_remaining_cushion():
    from real_engine import FundedVariant, run_funded
    v = FundedVariant("c", t1=2000, w1=0.5, t2=2000, w2=1.0, size2=1.0, need_win_days=0)
    # +2,000 -> take 1,000 (net 900), keep 1,000; +1,000 more -> balance 2,000, take it ALL (net 1,800)
    p1, p2, days, state = run_funded([2000.0, 1000.0, 1000.0], [0.0] * 3, 0, None, v, 4500, 0.9)
    assert (p1, p2, state) == (900.0, 1800.0, 2)
