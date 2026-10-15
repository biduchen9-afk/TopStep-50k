"""Compare fee paths under no_edge / positive_drift and find the breakeven Sharpe.

Usage: python run_comparison.py [--sharpes 0 0.5 1.5] [--policy cancel_and_reopen] [--sweep]
"""
from __future__ import annotations

import argparse

from pricing_config import BREACH_ACTIONS, load_config
from valuation_engine import breakeven_sharpe, value_path

COLS = ("path", "SR_gross", "SR_net", "p_mc", "p_bm", "E[tau]", "mu_att",
        "Cost_fund", "mu_ext", "EV_att", "EV_cycle", "days_fund")


def fmt_row(v) -> str:
    return (f"{v.path:<18}{v.sharpe_gross:>8.2f}{v.sharpe_net:>8.2f}{v.p_empirical:>8.3f}"
            f"{v.p_analytic:>8.3f}{v.e_tau:>8.1f}{v.mu_attempt:>9.0f}{v.cost_funded:>10.0f}"
            f"{v.mu_ext:>9.0f}{v.ev_attempt:>9.0f}{v.ev_cycle:>10.0f}{v.days_to_funded:>10.0f}")


def header() -> str:
    widths = (18, 8, 8, 8, 8, 8, 9, 10, 9, 9, 10, 10)
    return "".join(f"{c:<{w}}" if i == 0 else f"{c:>{w}}" for i, (c, w) in enumerate(zip(COLS, widths)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sharpes", type=float, nargs="+", default=[0.0, 0.5, 1.5])
    ap.add_argument("--policy", choices=BREACH_ACTIONS, default=None)
    ap.add_argument("--sweep", action="store_true", help="also solve breakeven Sharpe per path")
    args = ap.parse_args()

    cfg = load_config()
    if args.policy:
        cfg = cfg.with_policy(breach_action=args.policy)
    fp = cfg.farm_policy
    print(f"{cfg.firm} | breach_action={fp.breach_action} | n_sims={fp.n_simulations} | "
          f"sigma_daily=${cfg.market_process.sigma_daily:.0f} | friction=${cfg.friction.daily_cost:.2f}/day")

    for sr in args.sharpes:
        mode = "no_edge" if sr == 0 else "positive_drift"
        c = cfg.with_market(eval_mode=mode, sharpe_annual=sr)
        print(f"\n=== {mode} (gross Sharpe = {sr}) ===")
        print(header())
        for key in c.paths:
            v = value_path(c, key)
            print(fmt_row(v))
            if v.truncated:
                print(f"  ! {v.truncated} attempts hit max_days and were counted as failures")
        best = max((value_path(c, k) for k in c.paths), key=lambda v: v.ev_cycle)
        print(f"farm ({fp.max_concurrent_funded} funded slots, best path = {best.path}): "
              f"setup cost ${fp.max_concurrent_funded * best.cost_funded:,.0f}, "
              f"EV ${fp.max_concurrent_funded * best.ev_cycle:,.0f}")

    if args.sweep:
        print("\n=== Breakeven gross annual Sharpe (EV_attempt = 0) ===")
        for key in cfg.paths:
            print(f"{key:<18}{breakeven_sharpe(cfg, key):>8.3f}")


if __name__ == "__main__":
    main()
