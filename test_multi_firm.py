import numpy as np
from combine_policy import CombinePolicy
from contract_eval import CombineSizing
from farm_real import FarmSpec, run_farm
from firm_engine import topstep_150k
from multi_firm import FirmPlan, Policy, run_multifirm
from pricing_config import load_config
from real_engine import FundedPlan, Stream, rules_from_config


def _stream(seed, n=2200):
    rg = np.random.default_rng(seed)
    atr = rg.uniform(300, 700, n)
    pnl = rg.normal(14, 95, n)
    return Stream("t", np.arange(n).astype("datetime64[D]"), pnl, np.minimum(pnl, 0) - np.abs(rg.normal(0, 40, n)), atr, np.ones(n, dtype=bool))


def test_single_firm_unconstrained_matches_run_farm():
    cfg = load_config(); rules = rules_from_config(cfg, "no_activation_dll"); spec = topstep_150k("no_activation_dll")
    plan = FundedPlan("p", "standard", 3000.0, 0.5, 1.0, 0.0, 0.5, 1.0, 99, 0.4, 0.8)
    for seed in range(3):
        s = _stream(seed)
        ref = run_farm([s], CombineSizing("fixed", 9600.0), None, plan, cfg, rules, "no_activation_dll", 0, FarmSpec(max_evals=20, max_funded=5, p_est=1.0, refill="hyst"))
        out = run_multifirm([FirmPlan(spec, [s], 9600.0, plan)], Policy(reserve_cycles=None), 0)
        assert abs(out["net"] - ref["net"]) < 1e-6 and abs(out["payouts"] - ref["payouts"]) < 1e-6 and abs(out["fees"] - ref["fees"]) < 1e-6


def test_self_funding_never_injects_money_after_first_payout():
    spec = topstep_150k("no_activation_dll")
    plan = FundedPlan("p", "standard", 3000.0, 0.5, 1.0, 0.0, 0.5, 1.0, 99, 0.4, 0.8)
    s = _stream(1)
    out = run_multifirm([FirmPlan(spec, [s], 9600.0, plan)], Policy(k0=1, reserve_cycles=3.0), 0, log=True)
    ev = out["events"]
    pay_t = [e[0] for e in ev if e[2] == "payout"]
    assert pay_t, "synthetic stream should pay at least once"
    assert out["oop_total"] > 0
    # the out-of-pocket money is spent entirely before the first payout is received
    assert out["firms"][spec.name]["first_payout_t"] is not None


def test_group_aware_respects_caps_and_one_open_combine():
    from dataclasses import replace
    spec = topstep_150k("no_activation_dll")
    plan = FundedPlan("p", "standard", 2000.0, 0.5, 1.0, 0.0, 0.5, 1.0, 99, 1.2, 0.8)
    s1, s2 = _stream(1), _stream(2)
    fps = [FirmPlan(replace(spec, name="a", max_funded=3, max_evals=1), [s1], 9600.0, plan, firm="X", refill="all"), FirmPlan(replace(spec, name="b", max_funded=2, max_evals=1), [s2], 9600.0, plan, firm="X", refill="all")]
    out = run_multifirm(fps, Policy(reserve_cycles=3.0, group_aware=True, global_one_eval=True), 0, log=True)
    opens = [sum(e for e, _ in grp) for _, _, grp in out["daily"]]
    funded = [tuple(f for _, f in grp) for _, _, grp in out["daily"]]
    assert max(opens) <= 1
    assert max(f[0] for f in funded) <= 3 and max(f[1] for f in funded) <= 2
    assert out["oop_total"] > 0


def test_kelly_unit_is_causal_and_matches_formula():
    from kelly_sizing import HUGE, kelly_unit
    rng = np.random.default_rng(3)
    x = rng.normal(2.0, 30.0, 400)
    u = kelly_unit(x, window=100, min_obs=50)
    assert (u[:50] == HUGE).all()
    w = x[150:250]; mu, var = w.mean(), w.var()
    assert abs(u[250] - var / mu) < 1e-9
    y = x.copy(); y[250:] = 1e6                      # changing the present and future must not move the estimate at t=250
    assert abs(kelly_unit(y, window=100, min_obs=50)[250] - u[250]) < 1e-9
    assert (kelly_unit(-np.abs(x), window=100, min_obs=50) == HUGE).all()


def test_stage2_unit_changes_only_post_payout_sizing():
    from firm_engine import run_funded_firm, topstep_150k
    from real_engine import FundedPlan
    rng = np.random.default_rng(11)
    n = 400
    x = (rng.normal(8.0, 40.0, n)).tolist(); lo = (np.minimum(np.array(x), 0.0) - 5.0).tolist(); nd = [3] * n
    atr = [60.0] * n
    spec = topstep_150k().funded
    plan = FundedPlan("p", "x", 800.0, 0.5, 1.0, 0.0, 0.5, 1.0, 99, 1.2, 0.8)
    t1, t2, t3 = [], [], []
    base = run_funded_firm(spec, x, lo, nd, 0, plan, atr, t1)
    same = run_funded_firm(spec, x, lo, nd, 0, plan, atr, t2, list(atr))          # stage-2 unit identical -> identical result
    assert base == same and t1 == t2
    other = run_funded_firm(spec, x, lo, nd, 0, plan, atr, t3, [15.0] * n)       # 4x bigger post-payout size
    assert t3[0] == t1[0] and other[0][0] == base[0][0]                             # up to and including the first payout nothing changes
    assert other != base


def test_scale_none_and_neutral_schemes_reproduce_the_constant_size():
    from firm_engine import run_funded_firm, topstep_150k
    from real_engine import FundedPlan
    rng = np.random.default_rng(5)
    n = 500
    x = rng.normal(8.0, 40.0, n).tolist(); lo = (np.minimum(np.array(x), 0.0) - 5.0).tolist(); nd = [3] * n; atr = [600.0] * n
    spec = topstep_150k().funded
    base_plan = FundedPlan("p", "x", 1500.0, 0.5, 1.0, 0.0, 0.5, 1.0, 99, 1.2, 0.8)
    base = run_funded_firm(spec, x, lo, nd, 0, base_plan, atr)
    for sc in (("ladder", ((1e9, 9.0),)), ("ladder", ((0.0, 1.2),)), ("perf", 5, 1.0, 1.0), ("ramp", 20, 1.2)):
        got = run_funded_firm(spec, x, lo, nd, 0, FundedPlan("p", "x", 1500.0, 0.5, 1.0, 0.0, 0.5, 1.0, 99, 1.2, 0.8, sc), atr)
        assert got == base, sc
    for sc in (("ladder", ((300.0, 0.4),)), ("perf", 5, 1.5, 0.5), ("ramp", 20, 0.4)):
        assert run_funded_firm(spec, x, lo, nd, 0, FundedPlan("p", "x", 1500.0, 0.5, 1.0, 0.0, 0.5, 1.0, 99, 1.2, 0.8, sc), atr) != base, sc


def test_scaled_a_is_causal_and_follows_the_definitions():
    from firm_engine import _scaled_a
    assert _scaled_a(("ladder", ((500.0, 2.0), (1000.0, 2.5))), 0.4, 499.0, 3, []) == 0.4
    assert _scaled_a(("ladder", ((500.0, 2.0), (1000.0, 2.5))), 0.4, 500.0, 3, []) == 2.0
    assert _scaled_a(("ladder", ((500.0, 2.0), (1000.0, 2.5))), 0.4, 1200.0, 3, []) == 2.5
    assert _scaled_a(("ladder", ((500.0, 2.0),)), 0.4, 100.0, 3, []) == 0.4          # falling back below a milestone steps the size back down
    assert _scaled_a(("perf", 3, 1.5, 0.5), 1.0, 0.0, 2, [1.0, 1.0]) == 1.0          # fewer than L sessions -> unchanged
    assert _scaled_a(("perf", 3, 1.5, 0.5), 1.0, 0.0, 3, [5.0, -1.0, -1.0]) == 1.5
    assert _scaled_a(("perf", 3, 1.5, 0.5), 1.0, 0.0, 3, [9.0, -5.0, -5.0, -5.0]) == 0.5   # only the last 3 sessions count
    assert _scaled_a(("ramp", 10, 0.4), 1.2, 0.0, 0, []) == 0.4 and abs(_scaled_a(("ramp", 10, 0.4), 1.2, 0.0, 5, []) - 0.8) < 1e-12 and _scaled_a(("ramp", 10, 0.4), 1.2, 0.0, 99, []) == 1.2


def test_combine_and_after_payout_scaling_are_neutral_by_default_and_follow_definitions():
    from firm_engine import _eval_factor, _scaled_a, run_eval_firm, run_funded_firm, topstep_150k
    from real_engine import FundedPlan
    rng = np.random.default_rng(9)
    n = 600
    x = rng.normal(6.0, 40.0, n).tolist(); lo = (np.minimum(np.array(x), 0.0) - 5.0).tolist(); nd = [8] * n; atr = [300.0] * n
    ts = topstep_150k()
    base = run_eval_firm(ts.eval_phases, x, lo, nd, 0)
    assert run_eval_firm(ts.eval_phases, x, lo, nd, 0, ("const", 1.0)) == base
    assert run_eval_firm(ts.eval_phases, x, lo, nd, 0, ("ladder", ((1e9, 3.0),))) == base
    assert run_eval_firm(ts.eval_phases, x, lo, nd, 0, ("perf", 3, 1.0, 1.0)) == base
    assert run_eval_firm(ts.eval_phases, x, lo, nd, 0, ("ramp", 10, 1.0)) == base
    assert run_eval_firm(ts.eval_phases, x, lo, nd, 0, ("const", 0.25)) != base
    assert _eval_factor(("ladder", ((3000.0, 0.5), (6000.0, 0.25))), 2999.0, 4, []) == 1.0 and _eval_factor(("ladder", ((3000.0, 0.5), (6000.0, 0.25))), 6500.0, 4, []) == 0.25
    assert _eval_factor(("perf", 3, 1.5, 0.5), 0.0, 3, [1.0, -2.0, -2.0]) == 0.5 and _eval_factor(("perf", 3, 1.5, 0.5), 0.0, 2, [5.0, 5.0]) == 1.0
    assert _eval_factor(("ramp", 10, 0.5), 0.0, 0, []) == 0.5 and abs(_eval_factor(("ramp", 10, 0.5), 0.0, 5, []) - 0.75) < 1e-12
    spec = ts.funded
    plan = FundedPlan("p", "x", 800.0, 0.5, 1.0, 0.0, 0.5, 1.0, 99, 1.2, 0.8)
    b = run_funded_firm(spec, x, lo, nd, 0, plan, atr)
    for sc2 in (("ladder", ((1e9, 5.0),)), ("perf", 3, 1.0, 1.0), ("ramp", 10, 0.8), ("npay", ((99, 5.0),))):
        assert run_funded_firm(spec, x, lo, nd, 0, FundedPlan("p", "x", 800.0, 0.5, 1.0, 0.0, 0.5, 1.0, 99, 1.2, 0.8, None, sc2), atr) == b, sc2
    assert run_funded_firm(spec, x, lo, nd, 0, FundedPlan("p", "x", 800.0, 0.5, 1.0, 0.0, 0.5, 1.0, 99, 1.2, 0.8, None, ("ramp", 10, 0.1)), atr) != b
    t1, t2 = [], []
    p1 = run_funded_firm(spec, x, lo, nd, 0, plan, atr, t1)
    p2 = run_funded_firm(spec, x, lo, nd, 0, FundedPlan("p", "x", 800.0, 0.5, 1.0, 0.0, 0.5, 1.0, 99, 1.2, 0.8, None, ("ramp", 10, 0.1)), atr, t2)
    assert t1[0] == t2[0] and p1[0][0] == p2[0][0]                    # nothing changes up to and including the first payout
    assert _scaled_a(("npay", ((2, 1.2), (4, 1.6))), 0.8, 0.0, 3, [], 1) == 0.8 and _scaled_a(("npay", ((2, 1.2), (4, 1.6))), 0.8, 0.0, 3, [], 3) == 1.2 and _scaled_a(("npay", ((2, 1.2), (4, 1.6))), 0.8, 0.0, 3, [], 5) == 1.6
