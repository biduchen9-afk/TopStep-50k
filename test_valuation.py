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


# --- Real-data engine (Topstep-verified rules) -----------------------------------------------

def _rs(**kw):
    from real_engine import RuleSet
    return RuleSet(**kw)


def test_session_open_follows_us_dst_in_phnom_penh():
    """18:00 ET is 05:00 ICT in US summer time and 06:00 ICT in winter: never hard-code a local hour."""
    from datetime import datetime
    from market_data import session_open_phnom_penh
    assert session_open_phnom_penh(datetime(2024, 7, 10)).strftime("%H:%M") == "05:00"
    assert session_open_phnom_penh(datetime(2024, 12, 10)).strftime("%H:%M") == "06:00"


def test_combine_consistency_55_and_realtime_mll():
    from real_engine import run_combine
    z = [0.0] * 4
    ones = [1] * 4
    # best day 5,000 within 9,500 total breaks 55%? 5000/9500 = 52.6% -> passes on day 2
    assert run_combine([5000.0, 4500.0], [0.0, 0.0], [1, 1], 0, _rs()) == (1, 2)
    # best day 6,000 of 9,000 = 66% > 55%: the target rises, a further day is needed (9000 -> need 10,909)
    assert run_combine([6000.0, 3000.0, 2000.0], [0.0] * 3, [1] * 3, 0, _rs()) == (1, 3)
    # real-time MLL: closes +100 but the intraday low was -4,500 below the start -> breach that day
    assert run_combine([100.0], [-4500.0], [1], 0, _rs()) == (0, 1)
    # the old end-of-day rule would not have noticed
    assert run_combine([100.0], [-4500.0], [1], 0, _rs(intraday_mll=False)) == (-1, 1)


def test_combine_mll_locks_at_zero():
    from real_engine import run_combine
    # +4,500 then -4,400: floor locked at $0 -> a -4,400 day leaves bal=+100 (alive); another -150 breaches
    x = [4500.0, -4400.0, -150.0]
    assert run_combine(x, [0.0, -4400.0, -150.0], [1] * 3, 0, _rs())[0] == 0
    # peak +2,000 -> floor -2,500 (not yet locked); an intraday low of -4,500 (equity -2,500) touches it
    assert run_combine([2000.0, -4400.0], [0.0, -4500.0], [1, 1], 0, _rs()) == (0, 2)
    # a -4,400 low leaves equity at -2,400, above the floor: still alive
    assert run_combine([2000.0, -4400.0], [0.0, -4400.0], [1, 1], 0, _rs()) == (-1, 2)


def test_dll_liquidates_day_but_is_not_a_breach():
    from real_engine import run_combine
    r = _rs(dll=-3000.0)
    # intraday low -3,200, close +100: day booked at -3,000 and the account lives
    status, tau = run_combine([100.0, 100.0], [-3200.0, 0.0], [1, 1], 0, r)
    assert (status, tau) == (-1, 2)


def test_funded_payout_resets_mll_to_zero():
    from real_engine import FundedPlan, run_funded
    plan = FundedPlan("t", "standard", t1=2000, w1=0.5, t2=2000, w2=0.5)
    # +300 x 7 days -> balance 2,100 with 7 winning days; withdraw 50% = 1,050 gross (945 net); keep 1,050
    # day 8 +300 -> 1,350; day 9 intraday low -1,400 touches the NEW floor of $0 (equity -50) -> dead.
    # Before the reset the floor was -2,400 and the same day would have been survived.
    x = [300.0] * 8 + [-1400.0]
    lo = [0.0] * 8 + [-1400.0]
    p1, p2, days, state = run_funded(x, lo, [1] * 9, 0, _rs(), plan)
    assert round(p1, 6) == 945.0 and p2 == 0.0
    assert state == 1 and days == 9


def test_funded_standard_needs_five_winning_days_and_min_payout():
    from real_engine import FundedPlan, run_funded
    plan = FundedPlan("t", "standard", t1=2000, w1=0.5)
    # one +2,500 day then small days: balance is high on day 1 but only 1 winning day -> waits for 5
    x = [2500.0, 0.0, 0.0, 200.0, 200.0, 200.0, 200.0]
    p1, _, days, state = run_funded(x, [0.0] * 7, [1] * 7, 0, _rs(), plan)
    assert days == 7 and state == 2 and p1 > 0
    # consistency path: one monster day breaks the 40% rule; needs a diversified profit
    cons = FundedPlan("c", "consistency", t1=2000, w1=0.5)
    xs = [2500.0] + [400.0] * 11
    p1c, _, days_c, _ = run_funded(xs, [0.0] * 12, [1] * 12, 0, _rs(), cons)
    # best day 2,500 must be <= 40% of profit -> profit >= 6,250 -> reached on day 11 (2,500 + 10*400 = 6,500)
    assert days_c == 11 and p1c == 0.9 * min(0.5 * 6500.0, 6000.0)


def test_scaling_ladder_and_micro_caps():
    from real_engine import xfa_micro_cap
    assert [xfa_micro_cap(b) for b in (0, 1499, 1500, 2000, 3000, 4500)] == [30, 30, 40, 50, 100, 150]


def test_unresolved_combine_is_abandoned_after_a_year():
    from real_engine import run_combine
    assert run_combine([0.0] * 400, [0.0] * 400, [1] * 400, 0, _rs()) == (-1, 252)
