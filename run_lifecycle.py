"""3-year sequential lifecycle: mean/dispersion of net P&L and breakeven Sharpe."""
import argparse

import numpy as np

from lifecycle_simulator import PayoutRule, breakeven_sharpe_lifecycle, simulate_lifecycle
from pricing_config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=4000)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[600, 1000, 1500])
    ap.add_argument("--sharpes", type=float, nargs="+", default=[0.0, 0.3, 0.5, 0.8, 1.2])
    ap.add_argument("--trigger", type=float, default=3000.0)
    ap.add_argument("--path", default="no_activation_dll")
    args = ap.parse_args()

    cfg = load_config()
    rule = PayoutRule(trigger_profit=args.trigger)
    print(f"path={args.path} | payout trigger ${rule.trigger_profit:.0f}, {rule.payout_fraction:.0%} of balance, "
          f"cap ${rule.max_payout:.0f}, account closes after payout | 756 days | {args.paths} paths")
    print(f"{'sigma':>6}{'SR':>6}{'mean':>9}{'median':>9}{'std':>8}{'P(>0)':>7}{'p5':>8}{'p95':>8}"
          f"{'combines':>9}{'funded':>7}{'payouts':>8}{'fees':>7}")
    for sig in args.sigmas:
        for sr in args.sharpes:
            c = cfg.with_market(eval_mode="positive_drift", sharpe_annual=sr, sigma_daily=sig)
            r = simulate_lifecycle(c, args.path, args.paths, np.random.default_rng(7), payout=rule)
            t = r.total_pnl
            print(f"{sig:>6.0f}{sr:>6.1f}{t.mean():>9.0f}{np.median(t):>9.0f}{t.std():>8.0f}"
                  f"{(t > 0).mean():>7.2f}{np.percentile(t, 5):>8.0f}{np.percentile(t, 95):>8.0f}"
                  f"{r.n_combines.mean():>9.1f}{r.n_funded.mean():>7.1f}{r.n_payouts.mean():>8.1f}"
                  f"{r.fees.mean():>7.0f}")
        be = breakeven_sharpe_lifecycle(cfg.with_market(sigma_daily=sig), args.path,
                                        n_paths=args.paths, payout=rule)
        print(f"{'':>6}  -> breakeven gross Sharpe at sigma ${sig:.0f}: {be:.2f}")

if __name__ == "__main__":
    main()
