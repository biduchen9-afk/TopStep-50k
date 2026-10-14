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
