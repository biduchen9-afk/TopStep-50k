import numpy as np
from combine_policy import CombinePolicy
from contract_eval import CombineSizing
from farm_real import FarmSpec, run_farm
from pricing_config import load_config
from real_engine import FundedPlan, Stream, rules_from_config


def _stream(seed, n=1500):
    rg = np.random.default_rng(seed)
    atr = rg.uniform(300, 700, n)
    pnl = rg.normal(15, 90, n)
    return Stream("t", np.arange(n).astype("datetime64[D]"), pnl, np.minimum(pnl, 0) - np.abs(rg.normal(0, 40, n)), atr, np.ones(n, dtype=bool))


def test_never_more_than_five_funded_accounts():
    cfg = load_config(); rules = rules_from_config(cfg, "no_activation_dll")
    plan = FundedPlan("p", "standard", 4000.0, 0.5, 1.0, 0.0, 0.5, 1.0, 99, 0.4, 0.8)
    for seed in range(4):
        for refill in ("all", "hyst", "buffer"):
            for p_est in (1.0, 0.2):
                r = run_farm([_stream(seed)], CombineSizing("fixed", 9600.0), None, plan, cfg, rules, "no_activation_dll", 0,
                             FarmSpec(max_evals=20, max_funded=5, p_est=p_est, refill=refill), log=True)
                assert r["max_funded_seen"] <= 5
                assert max(f for _, _, f in r["daily"]) <= 5
                assert r["n_funded"] >= 1
