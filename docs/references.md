# Methodology references

Every analytic module in `src/topstep50k/` cites one or more of the
papers below. The convention in code comments is `[ref:<key>]`.

## Walk-forward, OOS evaluation, overfitting

| key | citation | id |
|---|---|---|
| `lopez_7reasons` | López de Prado, M. (2018). "The 10 Reasons Most Machine Learning Funds Fail." | SSRN 3104816 (extends SSRN 3031282) |
| `bailey_pbo` | Bailey, D.H., Borwein, J., López de Prado, M., Zhu, Q. (2014). "The Probability of Backtest Overfitting." | SSRN 2326253 |
| `lopez_pseudo` | Bailey, D.H., López de Prado, M. (2014). "Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance." | Notices of the AMS 61(5), 458-471; SSRN 2308659 |

## Performance attribution / inference

| key | citation | id |
|---|---|---|
| `bailey_psr` | Bailey, D.H., López de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier." Journal of Risk 15(2). | SSRN 1821643 |
| `bailey_dsr` | Bailey, D.H., López de Prado, M. (2014). "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality." | SSRN 2460551 |
| `mertens_psr_se` | Mertens, E. (2002). "Comments on Variance of the IID estimator in Lo (2002)." | working paper (used for PSR standard error closed form) |

## Resampling / Monte Carlo

| key | citation | id |
|---|---|---|
| `politis_romano` | Politis, D.N., Romano, J.P. (1994). "The Stationary Bootstrap." Journal of the American Statistical Association 89(428). | doi 10.2307/2290993 |
| `lopez_aml_ch11` | López de Prado, M. (2018). *Advances in Financial Machine Learning*, Ch. 11 ("Backtesting through Cross-Validation"), Ch. 13 ("Backtesting on Synthetic Data"). | book |

## Regime detection / HMM

| key | citation | id |
|---|---|---|
| `hamilton_1989` | Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle." Econometrica 57(2). | doi 10.2307/1912559 |
| `ang_bekaert_2002` | Ang, A., Bekaert, G. (2002). "Regime Switches in Interest Rates." Journal of Business & Economic Statistics 20(2). | doi 10.1198/073500102288618397 |
| `nystrup_2017` | Nystrup, P., Madsen, H., Lindström, E. (2017). "Long Memory of Financial Time Series and Hidden Markov Models with Time-Varying Parameters." Journal of Forecasting 36(8). | SSRN 2812613 |
| `nystrup_2018` | Nystrup, P., Boyd, S., Lindström, E., Madsen, H. (2019). "Multi-Period Portfolio Selection with Drawdown Control." Annals of Operations Research 282. | SSRN 3140370 |
| `rabiner_1989` | Rabiner, L.R. (1989). "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition." Proc. IEEE 77(2). | doi 10.1109/5.18626 (reference for forward-filter math) |

## Sizing / Kelly / EV gating

| key | citation | id |
|---|---|---|
| `kelly_1956` | Kelly, J.L. (1956). "A New Interpretation of Information Rate." Bell System Tech. J. 35. | doi 10.1002/j.1538-7305.1956.tb03809.x |
| `thorp_2006` | Thorp, E.O. (2006). "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market." Handbook of Asset and Liability Management, Vol. 1. | reprinted from working paper |
| `maclean_thorp_zionts` | MacLean, L.C., Thorp, E.O., Ziemba, W.T. (2010). "Good and Bad Properties of the Kelly Criterion." | SSRN 1559046 |
| `bailey_fractional` | Bailey, D.H., López de Prado, M. (2013). "Drawdown-Based Stop-Outs and the 'Triple Penance' Rule." Journal of Risk 18(2). | SSRN 2201302 |

## Correlation / multi-asset

| key | citation | id |
|---|---|---|
| `ledoit_wolf_2004` | Ledoit, O., Wolf, M. (2004). "Honey, I Shrunk the Sample Covariance Matrix." Journal of Portfolio Management 30(4). | doi 10.3905/jpm.2004.110 (shrinkage covariance) |
| `lopez_hrp` | López de Prado, M. (2016). "Building Diversified Portfolios that Outperform Out-of-Sample." Journal of Portfolio Management 42(4). | SSRN 2708678 (Hierarchical Risk Parity) |

## TopStep $50K rule sources

See `docs/rules_sources.md` — citations are help-center articles on
help.topstep.com, not academic papers. Verification protocol there.

## Citation rules in code

- Every new analytic module's docstring lists the `[ref:<key>]` it
  implements or extends.
- A test must demonstrate the property the paper claims (e.g. PSR
  monotonicity in N, PBO sanity check on shuffled returns).
- If a method intentionally diverges from the cited paper, the
  divergence is documented inline AND in this file's notes column.
