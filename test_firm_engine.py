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
        spec = topstep_150k(fee_path, payout_path)
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
