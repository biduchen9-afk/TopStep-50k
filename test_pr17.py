"""PR17 additions: tick-based liquidation overshoot, price factor, cohort accounting switch, cost re-basing arithmetic."""
import dataclasses
import numpy as np
from firm_engine import FundedPlan, run_funded_firm, run_eval_firm, topstep_150k


def _plan():
    return FundedPlan("p", "standard", 0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 99, 0.4, 0.8)


def test_tick_slip_is_neutral_by_default_and_scales_with_contracts_and_fx():
    spec = topstep_150k("no_activation_dll", "standard", rules="v1")
    f0 = spec.funded
    f1 = dataclasses.replace(f0, dll_slip_ticks=2.0)
    n = 10
    x = [400.0] * 6 + [-5000.0] + [0.0] * 3
    lo = list(x); nd = [10] * n; atr = [1.0] * n
    a = run_funded_firm(f0, x, lo, nd, 0, _plan(), atr)
    b = run_funded_firm(f1, x, lo, nd, 0, _plan(), atr)
    assert a[3] >= b[3]                                   # overshoot never helps
    half = run_funded_firm(f1, x, lo, nd, 0, _plan(), atr, fx=[0.5] * n)
    assert a[3] - b[3] >= a[3] - half[3] >= 0.0           # a smaller price factor means a smaller overshoot


def test_eval_tick_slip_neutral_default():
    spec = topstep_150k("no_activation_dll", "standard", rules="v1")
    x = [100.0] * 30; lo = list(x); nd = [5] * 30
    assert run_eval_firm(spec.eval_phases, x, lo, nd, 0) == run_eval_firm(spec.eval_phases, x, lo, nd, 0, None, None)


def test_rebase_arithmetic_and_identity_at_dev_costs():
    import retest as T, retest_cost as C
    D = T.load(); ST = D["ST"]
    same = C.rebase(ST, 1.0, to_today=False)
    for g in ST:
        assert np.allclose(same[g].pnl, ST[g].pnl) and np.allclose(same[g].low, ST[g].low)
    lo = C.rebase(ST, 1.0, to_today=True)
    for g in ST:
        ok = ST[g].ok
        assert np.all(lo[g].pnl[ok] >= ST[g].pnl[ok] - 1e-9)      # cheaper costs can only raise P&L
        assert np.all(lo[g].low <= 1e-12)
    assert abs(C.modelled_cost_usd() - 1.72) < 1e-12


def test_stop_buy_at_stops_purchases():
    import retest as T
    from multi_firm import Policy, run_multifirm
    D = T.load(); ST, ref, n_dev = D["ST"], D["ref"], D["n_dev"]
    fps = T.farm_plans(ST, [("noise", 3, T.BASE), ("gap", 2, T.BASE)])
    i0 = T.farm_starts(ref, n_dev)[40]
    full = run_multifirm(fps, Policy(reserve_cycles=None, global_one_eval=True), i0, i0 + 756)
    cut = run_multifirm(fps, Policy(reserve_cycles=None, global_one_eval=True, stop_buy_at=i0 + 300), i0, i0 + 756)
    nev = lambda r: sum(v["n_evals"] for v in r["firms"].values())
    assert nev(cut) < nev(full)
    same = run_multifirm(fps, Policy(reserve_cycles=None, global_one_eval=True, stop_buy_at=i0 + 756 + 10), i0, i0 + 756)
    assert same["net"] == full["net"]
