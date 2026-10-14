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


def test_breach_policies_billing_clock():
    from pricing_config import load_config
    from chain_real import combine_chain
    cfg = load_config()
    rs = _rs()
    # sessions 0 and 21 start attempts; session 0 breaches at once (low -4,500), 21-22 are two +5,000 days (pass)
    n = 60
    x, lo = [0.0] * n, [0.0] * n
    lo[0] = -4500.0
    x[21], x[22] = 5000.0, 5000.0
    nd = [1] * n
    sub = cfg.paths["standard"].monthly_cost
    act = cfg.paths["standard"].activation_fee
    # wait_for_rebill: idle 1..20, renewal on session 21 grants the reset -> two bills, attempt starts at 21
    sp, fees, att, _ = combine_chain(x, lo, nd, 0, n, rs, cfg, "standard", "wait_for_rebill")
    assert (sp, att, fees) == (22, 2, 2 * sub + act)
    # cancel_and_reopen: new subscription on session 1; that attempt passes only if it sees +5,000 days -> shift the pass to 1-2
    x2, lo2 = [0.0] * n, [0.0] * n
    lo2[0] = -4500.0
    x2[1], x2[2] = 5000.0, 5000.0
    sp, fees, att, _ = combine_chain(x2, lo2, nd, 0, n, rs, cfg, "standard", "cancel_and_reopen")
    assert (sp, att, fees) == (2, 2, 2 * sub + act)
    # reset fee equals the subscription fee on every path, so 'immediate_reset' == cancel-and-reopen
    assert combine_chain(x2, lo2, nd, 0, n, rs, cfg, "standard", "immediate_reset") == combine_chain(x2, lo2, nd, 0, n, rs, cfg, "standard", "cancel_and_reopen")


def test_repeated_payouts_until_death():
    from real_engine import FundedPlan, run_funded_detail
    plan = FundedPlan("m", "standard", t1=2000, w1=0.5, t2=1500, w2=0.5, max_payouts=3)
    # steady +300/day: payouts happen repeatedly, then a crash kills the account at the $0 floor
    x = [300.0] * 40 + [-9000.0]
    lo = [0.0] * 40 + [-9000.0]
    pays, days, state, _ = run_funded_detail(x, lo, [1] * 41, 0, _rs(), plan)
    assert len(pays) == 3 and state == 2        # closed by max_payouts before the crash
    plan_inf = FundedPlan("m", "standard", t1=2000, w1=0.5, t2=1500, w2=0.5, max_payouts=99)
    pays, days, state, _ = run_funded_detail(x, lo, [1] * 41, 0, _rs(), plan_inf)
    assert len(pays) > 3 and state == 1         # keeps paying until the crash


# --- Academic sleeves ---------------------------------------------------------------------

def _day(prices_open, drift_after=0.0, n=1320):
    """A synthetic session: flat at `prices_open` until 09:30 ET (minute 930), then drifting by drift_after per minute."""
    import numpy as np
    smin = np.arange(n)
    c = np.full(n, prices_open, dtype=float)
    c[930:] = prices_open + drift_after * (smin[930:] - 930)
    o = np.r_[c[0], c[:-1]]
    h, l = np.maximum(o, c) + 0.1, np.minimum(o, c) - 0.1
    return o, h, l, c, np.full(n, 100.0), smin


def test_intraday_momentum_trades_last_half_hour_in_direction_of_first_half_hour():
    from academic_strategies import intraday_mom_session
    o, h, l, c, v, m = _day(100.0, drift_after=0.01)        # rises after the open -> first half hour is positive vs prev close
    pnl, low, trades = intraday_mom_session(o, h, l, c, v, m, prev_close=99.0, atr=2.0, cost_pts=0.0)
    assert trades == 1 and pnl > 0                           # long into a still-rising last half hour
    pnl2, _, t2 = intraday_mom_session(o, h, l, c, v, m, prev_close=99.0, atr=2.0, cost_pts=0.0, allow=-1)
    assert t2 == 0 and pnl2 == 0                             # a short-only gate blocks a long signal


def test_noise_area_breakout_long_and_gate():
    import numpy as np
    from academic_strategies import noise_area_session
    o, h, l, c, v, m = _day(100.0, drift_after=0.02)
    sigma = np.full(12, 0.001)                               # the usual noise is 0.1%; a steady drift breaks out of it
    pnl, low, trades = noise_area_session(o, h, l, c, v, m, prev_close=100.0, sigma=sigma, cost_pts=0.0)
    assert trades >= 1 and pnl > 0
    assert noise_area_session(o, h, l, c, v, m, prev_close=100.0, sigma=sigma, cost_pts=0.0, allow=-1)[2] == 0


# --- Dynamic (cushion-proportional) sizing -----------------------------------------------------

def test_combine_dynamic_sizing_scales_with_cushion():
    from real_engine import run_combine
    n = 10
    x, lo = [1.0] * n, [0.0] * n                  # +$1 per micro per day, no intraday dip
    atr = [100.0] * n
    # risk = 1.0 x 4,500 cushion at $100 per micro -> 45 micros; each day adds 45 x $1
    status, tau = run_combine(x, lo, [1] * n, 0, _rs(profit_target=100.0), dyn_a=1.0, atr=atr)
    assert (status, tau) == (1, 3)                # 45 + 45 + 45 = 135 >= 100 on day 3 (best day 45 <= 55% of 135)


def test_funded_dynamic_risk_shrinks_after_payout():
    from real_engine import FundedPlan, run_funded_detail
    plan = FundedPlan("d", "standard", t1=500, w1=0.5, t2=0.0, w2=0.5, max_payouts=99, a1=0.2, a2=0.2)
    n = 400
    x = [1.0] * n                                   # +$1 per micro per day
    pays, days, state, bal = run_funded_detail(x, [0.0] * n, [1] * n, 0, _rs(), plan, atr=[10.0] * n)
    assert len(pays) >= 2 and state == 0            # keeps compounding small payouts; never dies on a steady series


def test_farm_respects_account_caps():
    import numpy as np
    from pricing_config import load_config
    from contract_eval import CombineSizing
    from farm_real import FarmSpec, run_farm
    from real_engine import FundedPlan, Stream, rules_from_config
    cfg = load_config()
    rng = np.random.default_rng(1)
    n = 1500
    pnl = rng.normal(0, 400, n)
    s = Stream("rand", np.arange(n).astype("datetime64[D]"), pnl, np.minimum(pnl, 0) - 100, np.full(n, 500.0), np.ones(n, bool))
    plan = FundedPlan("p", "standard", 1500, 0.5, 1.0, 0.0, 0.5, 1.0, 99, 0.2, 0.2)
    out = run_farm([s], CombineSizing("fixed", 3000), None, plan, cfg, rules_from_config(cfg, "no_activation_dll"), "no_activation_dll", 0, FarmSpec())
    assert out["max_funded_seen"] <= 5 and out["max_evals_seen"] <= 20 and out["n_evals"] > 0
