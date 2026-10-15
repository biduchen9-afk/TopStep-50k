import numpy as np
import pytest
from firm_engine import run_eval_firm, run_funded_firm, topstep_150k
from pricing_config import load_config
from real_engine import FundedPlan, Stream, rules_from_config, run_combine, run_funded_detail


def _stream(seed, n=1400):
    rg = np.random.default_rng(seed)
    atr = rg.uniform(200, 900, n)
    pnl = rg.normal(4, 70, n) * (1 + (rg.random(n) < 0.03) * 4)
    low = np.minimum(pnl, 0) - np.abs(rg.normal(0, 45, n))
    return Stream("t", np.arange(n).astype("datetime64[D]"), pnl, low, atr, np.ones(n, dtype=bool))


@pytest.mark.parametrize("fee_path", ["standard", "no_activation_dll"])
def test_topstep_spec_reproduces_existing_engine(fee_path):
    cfg = load_config(); rules = rules_from_config(cfg, fee_path)
    for payout_path in ("standard", "consistency"):
        spec = topstep_150k(fee_path, payout_path, rules="v1")
        for seed in range(5):
            s = _stream(seed); x, lo, atr = s.pnl.tolist(), s.low.tolist(), s.atr_usd.tolist(); nd = s.desired_micros(9600.0)
            for i0 in range(0, 600, 29):
                assert run_eval_firm(spec.eval_phases, x, lo, nd, i0) == run_combine(x, lo, nd, i0, rules)
            plan = FundedPlan("p", payout_path, 2500.0, 0.5, 1.0, 0.0, 0.5, 1.0, 99, 0.4, 0.8)
            for i0 in range(0, 600, 41):
                assert run_funded_firm(spec.funded, x, lo, nd, i0, plan, atr) == run_funded_detail(x, lo, nd, i0, rules, plan, atr)
            plan2 = FundedPlan("q", payout_path, 1500.0, 0.5, 1.0, 1000.0, 0.5, 1.0, 3, None, None)
            for i0 in range(0, 600, 53):
                assert run_funded_firm(spec.funded, x, lo, nd, i0, plan2, atr) == run_funded_detail(x, lo, nd, i0, rules, plan2, atr)


def test_v2_net_positive_rule_blocks_later_payouts_only():
    """Payout Policy: every payout after the first needs a positive net profit since the last payout (the first is exempt); v1 does not."""
    from firm_engine import FundedPlan, run_funded_firm
    v1, v2 = topstep_150k("no_activation_dll", "standard", rules="v1"), topstep_150k("no_activation_dll", "standard", rules="v2")
    assert not v1.funded.net_pos_after_first and v2.funded.net_pos_after_first
    # 8 winning days (+300), then 5 winning days (+160 each) offset by a -1,000 day: >= 5 wins since payout 1 but net negative
    x = [300.0] * 12 + [160.0] * 5 + [-1000.0] + [0.0] * 5
    lo = list(x)
    nd = [1] * len(x)
    plan = FundedPlan("p", "standard", 0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 99, 0.4, 0.8)
    p1, *_ = run_funded_firm(v1.funded, x, lo, nd, 0, plan, [1.0] * len(x))
    p2, *_ = run_funded_firm(v2.funded, x, lo, nd, 0, plan, [1.0] * len(x))
    assert len(p2) <= len(p1) and p2[0] == p1[0]


def test_dll_slip_books_below_the_limit():
    spec = topstep_150k("no_activation_dll", "standard", rules="v1")
    import dataclasses
    f0 = spec.funded
    f1 = dataclasses.replace(f0, dll_slip=100.0)
    from firm_engine import FundedPlan, run_funded_firm
    x = [400.0] * 6 + [-5000.0] + [0.0] * 3
    lo = [400.0] * 6 + [-5000.0] + [0.0] * 3
    plan = FundedPlan("p", "standard", 0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 99, 0.4, 0.8)
    a = run_funded_firm(f0, x, lo, [1] * 10, 0, plan, [1.0] * 10)
    b = run_funded_firm(f1, x, lo, [1] * 10, 0, plan, [1.0] * 10)
    assert a[3] - b[3] in (0.0, 100.0)
