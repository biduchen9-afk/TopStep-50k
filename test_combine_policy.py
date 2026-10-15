import numpy as np
from combine_policy import CombinePolicy, run_combine_policy
from pricing_config import load_config
from real_engine import Stream, rules_from_config, run_combine


def _stream(seed, n=900):
    rg = np.random.default_rng(seed)
    atr = rg.uniform(200, 900, n)
    pnl = rg.normal(3, 60, n)
    low = np.minimum(pnl, 0) - np.abs(rg.normal(0, 40, n))
    return Stream("t", np.arange(n).astype("datetime64[D]"), pnl, low, atr, np.ones(n, dtype=bool))


def test_neutral_policy_equals_fixed_risk_combine():
    cfg = load_config()
    for path in ("standard", "no_activation_dll"):
        rules = rules_from_config(cfg, path)
        for seed in range(6):
            s = _stream(seed)
            nd = s.desired_micros(9600.0)
            pol = CombinePolicy(R=9600.0)
            for i0 in range(0, 400, 17):
                assert run_combine_policy(s.pnl.tolist(), s.low.tolist(), s.atr_usd.tolist(), i0, rules, pol) == run_combine(s.pnl.tolist(), s.low.tolist(), nd, i0, rules)


def test_cushion_cap_never_increases_risk():
    cfg = load_config()
    rules = rules_from_config(cfg, "standard")
    s = _stream(1)
    capped = CombinePolicy(R=9600.0, cap_c=1.0)
    free = CombinePolicy(R=9600.0)
    nb = sum(run_combine_policy(s.pnl.tolist(), s.low.tolist(), s.atr_usd.tolist(), i, rules, capped)[0] == 0 for i in range(0, 500, 5))
    nf = sum(run_combine_policy(s.pnl.tolist(), s.low.tolist(), s.atr_usd.tolist(), i, rules, free)[0] == 0 for i in range(0, 500, 5))
    assert nb <= nf
