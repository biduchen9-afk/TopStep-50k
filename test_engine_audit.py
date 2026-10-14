"""Audit additions: tests that fail if the real-data engine gets money, fees or rule direction wrong.
Every test here was checked against deliberately broken copies of the engine (see audit notes)."""
import numpy as np
import pytest

from pricing_config import load_config


def _rs(**kw):
    from real_engine import RuleSet
    return RuleSet(**kw)


def test_dll_day_is_booked_at_the_limit_not_at_the_close():
    from real_engine import run_combine
    # day 1: low -3,200, close +100 -> booked -3,000. Day 2 low -1,500 then touches the MLL (-4,500 from -3,000): breach.
    # (If day 1 were booked at +100 the account would still be alive on day 2.)
    assert run_combine([100.0, -1400.0], [-3200.0, -1500.0], [1, 1], 0, _rs(dll=-3000.0)) == (0, 2)
    # EOD-only MLL: the DLL still caps day 1 at -3,000; day 2 closes -1,600 -> -4,600 <= -4,500
    assert run_combine([100.0, -1600.0], [-3200.0, -1600.0], [1, 1], 0, _rs(dll=-3000.0, intraday_mll=False)) == (0, 2)


def test_combine_mll_locks_once_peak_exceeds_the_mll():
    from real_engine import run_combine
    x, lo = [6000.0, -4000.0], [0.0, -5000.0]
    # peak 6,000 -> the floor would be +1,500 if it trailed forever; it locks at 0, so equity +1,000 is alive
    assert run_combine(x, lo, [1, 1], 0, _rs()) == (-1, 2)
    assert run_combine(x, lo, [1, 1], 0, _rs(combine_mll_locks=False)) == (0, 2)


def test_funded_floor_trails_eod_high_and_is_realtime():
    from real_engine import FundedPlan, run_funded_detail
    plan = FundedPlan("t", "standard", t1=1e9)                    # never pays
    # EOD high +3,000 -> floor -1,500. A low of -4,400 (equity -1,400) is alive; -4,500 (equity -1,500) is dead.
    assert run_funded_detail([3000.0, 0.0], [0.0, -4400.0], [1, 1], 0, _rs(), plan)[1:3] == (2, 0)
    assert run_funded_detail([3000.0, 0.0], [0.0, -4500.0], [1, 1], 0, _rs(), plan)[1:3] == (2, 1)


def test_funded_ladder_limits_contracts():
    from real_engine import FundedPlan, run_funded_detail
    plan = FundedPlan("t", "standard", t1=1e9)
    # +$10 per micro, 200 micros wanted: 30 micros until bal 1,500, then 40, 50 (>=2,000), 100 (>=3,000), 150 (>=4,500)
    pays, days, state, bal = run_funded_detail([10.0] * 12, [0.0] * 12, [200] * 12, 0, _rs(), plan)
    assert (days, state, bal) == (12, 0, 6800.0)


def test_payout_is_half_the_balance_capped_and_deducted():
    from real_engine import FundedPlan, run_funded_detail
    plan = FundedPlan("t", "standard", t1=2000, w1=0.9)             # asks for 90%: Topstep allows 50%
    pays, _, state, bal = run_funded_detail([1000.0] * 5, [0.0] * 5, [1] * 5, 0, _rs(), plan)
    assert [round(p, 6) for p in pays] == [2250.0] and bal == 2500.0 and state == 2
    pays, _, _, bal = run_funded_detail([4000.0] * 5, [0.0] * 5, [1] * 5, 0, _rs(), plan)   # 50% of 20,000 > cap 5,000
    assert [round(p, 6) for p in pays] == [4500.0] and bal == 15000.0


def test_funded_win_day_threshold_is_150():
    from real_engine import FundedPlan, run_funded_detail
    plan = FundedPlan("t", "standard", t1=0.0)
    pays, days, state, _ = run_funded_detail([149.0] * 10, [0.0] * 10, [1] * 10, 0, _rs(), plan)
    assert pays == [] and state == 0                                # $149 days never count as winning days
    pays, days, state, _ = run_funded_detail([150.0] * 10, [0.0] * 10, [1] * 10, 0, _rs(), plan)
    assert days == 5 and len(pays) == 1


def test_funded_dynamic_a2_applies_after_the_first_payout():
    from real_engine import FundedPlan, run_funded_detail
    n = 80

    def second_payout_day(a2):
        plan = FundedPlan("d", "standard", t1=500, w1=0.5, t2=2500, w2=0.5, max_payouts=2, a1=0.1, a2=a2)
        pays, days, state, _ = run_funded_detail([10.0] * n, [0.0] * n, [1] * n, 0, _rs(), plan, atr=[5.0] * n)
        assert len(pays) == 2 and state == 2
        return days
    assert second_payout_day(0.3) < second_payout_day(0.1)        # a larger post-payout multiplier reaches the 2nd payout sooner


def test_combine_outcomes_fee_accounting():
    from real_engine import combine_outcomes
    cfg = load_config()
    n = 120
    x, lo = [0.0] * n, [0.0] * n
    lo[24] = -4500.0                                                # start 0: breach on session 25 -> 2 bills
    x[40], x[41] = 5000.0, 5000.0                                   # start 30: passes after 12 sessions -> 1 bill + activation
    out = combine_outcomes(x, lo, [1] * n, cfg, "standard", _rs(), np.array([0, 30, 100]))
    assert list(out["status"]) == [0, 1, -1] and list(out["tau"]) == [25, 12, 20]
    assert list(out["fees"]) == [2 * 199.0, 199.0 + 149.0, 199.0]   # unfinished attempt still pays; activation only on a pass
    out = combine_outcomes(x, lo, [1] * n, cfg, "no_activation_dll", _rs(dll=-3000.0), np.array([30]))
    assert list(out["fees"]) == [199.0]


def test_chain_fees_by_hand_all_policies():
    from chain_real import combine_chain
    cfg = load_config()
    n = 200
    x, lo = [0.0] * n, [0.0] * n
    lo[24] = -4500.0
    lo[25 + 42] = -4500.0                                           # attempts: tau 25 (2 bills), tau 43 (3 bills), then pass
    x[68] = x[69] = 5000.0                                          # pass: 1 bill + 149 activation
    exp = 6 * 199.0 + 149.0
    for pol in ("cancel_and_reopen", "immediate_reset", "immediate_reset_keep"):
        assert combine_chain(x, lo, [1] * n, 0, n, _rs(), cfg, "standard", pol) == (69, exp, 3, 69), pol


def test_replay_detail_charges_every_fee():
    from chain_real import replay_detail
    from real_engine import FundedPlan, Stream, rules_from_config
    cfg = load_config()
    n = 200
    pnl = np.zeros(n); low = np.zeros(n)
    low[24] = -4500.0; low[67] = -4500.0
    pnl[68] = pnl[69] = 5000.0
    s = Stream("s", np.arange(n).astype("datetime64[D]"), pnl, low, np.full(n, 100.0), np.ones(n, bool))
    plan = FundedPlan("p", "standard", t1=1e9)
    out = replay_detail(s, 100.0, s, 100.0, plan, cfg, rules_from_config(cfg, "standard"), "standard", 0)
    assert out["fees"] == 6 * 199.0 + 149.0 and out["attempts"] == 3 and out["funded"] == 1


def test_replay_detail_restarts_after_a_funded_death():
    from chain_real import replay_detail
    from real_engine import FundedPlan, Stream, rules_from_config
    cfg = load_config()
    n = 400
    pnl = np.zeros(n); low = np.zeros(n)
    pnl[0] = pnl[1] = 5000.0                    # Combine 1 passes on session 1; funded runs sessions 2-3
    low[3] = -4500.0                            # funded account dies on session 3 (floor -4,500)
    pnl[4] = pnl[5] = 5000.0                    # Combine 2 starts on session 4 and passes on session 5
    s = Stream("s", np.arange(n).astype("datetime64[D]"), pnl, low, np.full(n, 100.0), np.ones(n, bool))
    out = replay_detail(s, 100.0, s, 100.0, FundedPlan("p", "standard", t1=1e9), cfg, rules_from_config(cfg, "standard"), "standard", 0)
    assert (out["attempts"], out["funded"], out["died"]) == (2, 2, 1) and out["fees"] == 2 * (199.0 + 149.0)


def test_contract_metrics_net_is_payouts_minus_fees():
    from contract_eval import CombineSizing, contract_metrics
    from real_engine import FundedPlan, Stream, rules_from_config
    from reference_impl import ref_funded
    cfg = load_config()
    n = 200
    s = Stream("s", np.arange(n).astype("datetime64[D]"), np.full(n, 5000.0), np.zeros(n), np.full(n, 100.0), np.ones(n, bool))
    plan = FundedPlan("p", "standard", t1=2000, w1=0.5, t2=0.0, w2=0.5, max_payouts=99)
    starts = np.arange(0, 50)
    m = contract_metrics(s, s, CombineSizing("fixed", 100.0), plan, None, cfg, rules_from_config(cfg, "standard"), starts,
                         np.full(len(starts), 2020), path="standard")
    exp = [sum(ref_funded([5000.0] * n, [0.0] * n, [1] * n, None, i + 2, t1=2000, t2=0.0, max_payouts=99)[0]) for i in starts]
    assert m["p_pass"] == 1.0 and m["cost_funded"] == 199.0 + 149.0
    assert m["extraction_per_funded"] == pytest.approx(np.mean(exp))
    assert m["ev_att"] == pytest.approx(np.mean(exp) - 348.0)


def test_independent_reference_implementation_agrees():
    """Differential test against reference_impl.py (a separate equity-level state machine)."""
    from real_engine import FundedPlan, run_combine, run_funded_detail
    from reference_impl import ref_combine, ref_funded
    rng = np.random.default_rng(2026)
    for _ in range(400):
        n = int(rng.integers(30, 500))
        sig = float(rng.choice([2, 10, 40, 80]))
        x = rng.normal(sig / 12, sig, n)
        x = np.where(rng.random(n) < .03, 3 * x, x)
        lo = np.minimum(np.minimum(x, 0), x - np.abs(rng.normal(0, sig, n)))
        atr = np.abs(rng.normal(3 * sig, sig / 2, n)) + 1
        nd = rng.integers(1, 40, n).tolist()
        x, lo, atr = x.tolist(), lo.tolist(), atr.tolist()
        dll = None if rng.random() < .5 else -3000.0
        intr = bool(rng.random() < .8)
        dyn = None if rng.random() < .5 else float(rng.choice([.25, .75, 1.5]))
        i0 = int(rng.integers(0, n // 2))
        assert run_combine(x, lo, nd, i0, _rs(dll=dll, intraday_mll=intr), dyn, atr) == ref_combine(x, lo, nd, atr, i0, dll=dll, a=dyn, intraday=intr)
        path = "standard" if rng.random() < .5 else "consistency"
        a1 = None if rng.random() < .5 else float(rng.choice([.05, .2, .4]))
        a2 = None if a1 is None else float(rng.choice([.05, .4]))
        t2 = None if rng.random() < .3 else float(rng.choice([0., 1500.]))
        plan = FundedPlan("p", path, float(rng.choice([1500., 3000.])), .5, 1., t2, .5, 1., int(rng.choice([2, 99])), a1, a2)
        a = run_funded_detail(x, lo, nd, i0, _rs(dll=dll, intraday_mll=intr), plan, atr)
        b = ref_funded(x, lo, nd, atr, i0, path=path, t1=plan.t1, t2=t2, max_payouts=plan.max_payouts, a1=a1, a2=a2, dll=dll, intraday=intr)
        assert len(a[0]) == len(b[0]) and np.allclose(a[0], b[0]) and a[1:3] == b[1:3] and abs(a[3] - b[3]) < 1e-6


def test_farm_fee_accounting_and_slot_limits(monkeypatch):
    import farm_real
    from contract_eval import CombineSizing
    from real_engine import FundedPlan, Stream, rules_from_config
    cfg = load_config()
    rng = np.random.default_rng(3)
    n = 1200
    pnl = rng.normal(40, 300, n)
    s = Stream("rand", np.arange(n).astype("datetime64[D]"), pnl, np.minimum(pnl, 0) - 50, np.full(n, 500.0), np.ones(n, bool))
    calls = {"c": [], "f": []}
    rc, rf = farm_real.run_combine, farm_real.run_funded_detail

    def spy_c(*a):
        r = rc(*a); calls["c"].append((a[3], r)); return r

    def spy_f(*a):
        r = rf(*a); calls["f"].append((a[3], r)); return r
    monkeypatch.setattr(farm_real, "run_combine", spy_c)
    monkeypatch.setattr(farm_real, "run_funded_detail", spy_f)
    for path in ("standard", "no_activation_dll"):
        calls["c"].clear(); calls["f"].clear()
        spec = farm_real.FarmSpec(max_evals=20, max_funded=5, p_est=0.0, starts_per_day=3)
        plan = FundedPlan("p", "standard", 1500, 0.5, 1.0, 0.0, 0.5, 1.0, 99, 0.2, 0.2)
        out = farm_real.run_farm([s], CombineSizing("fixed", 3000), None, plan, cfg, rules_from_config(cfg, path), path, 0, spec)
        sub, act = cfg.paths[path].monthly_cost, cfg.paths[path].activation_fee
        fees = sum((1 + (tau - 1) // 21) * sub for _, (st, tau) in calls["c"]) + len(calls["f"]) * act
        assert out["fees"] == pytest.approx(fees) and out["n_evals"] == len(calls["c"]) and out["n_funded"] == len(calls["f"]) > 0
        assert out["payouts"] == pytest.approx(sum(sum(r[0]) for _, r in calls["f"]))
        ev, fu = np.zeros(n + 400), np.zeros(n + 400)
        for t, (st, tau) in calls["c"]:
            ev[t:t + tau] += 1
        for t, (pays, days, state, bal) in calls["f"]:
            fu[t:t + days] += 1
        assert ev.max() == 20 and fu.max() == 5                    # the aggressive spec really hits both limits and never exceeds them
        assert len(calls["f"]) <= sum(1 for _, (st, tau) in calls["c"] if st == 1)   # each activation needs a passed Combine
