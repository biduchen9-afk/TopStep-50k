# TopStep-50k

Audit-first backtesting engine for the TopStep $50K Trading Combine.

## Status — pause point 2026-05-17

Audit-first foundation is complete and green (**57 unit + integration tests passing**).
Real-data integration is blocked on file delivery (see "Data" below).

## What's built

| Module | What it does |
|---|---|
| `rules/topstep.py` | $50K Combine rule encoders. Trailing MLL state machine, daily loss limit (soft), profit target, position size cap, consistency rule. Decimal arithmetic, every numeric constant cited in `docs/rules_sources.md`. |
| `engine/clock.py` | Monotonic clock with `assert_visible()` guard. Raises `LookAheadError` on any access strictly newer than `now()`. |
| `engine/types.py` | Bar (tz-aware required), Order, Fill, Instrument (tick-accurate PnL math). |
| `engine/ledger.py` | Average-cost position book, mark-to-market equity, day boundary handling. |
| `engine/backtest.py` | Event-driven loop. Strategies emit **target positions** on bar `t`; orders fill at bar `t+1`'s open — structural no-look-ahead. Hard breach flattens + halts, soft breach flattens + blocks new entries for the session. |
| `data/source.py` | `DataSource` protocol + `InMemoryBarSource`. `history()` calls `clock.assert_visible()` on every bar it returns. |
| `data/loaders.py` | Schema-sniffing CSV/TXT loader (NinjaTrader `YYYYMMDD HHMMSS;O;H;L;C;V` + ISO formats). |
| `audit/log.py` | Append-only `AuditLog`. In-memory or streaming JSONL. Every state mutation produces a structured event keyed by `clock.now()`. |
| `analysis/stats.py` | Sharpe, Sortino (downside-only), max drawdown ($ + %), profit factor, win rate, `performance()` rollup. |
| `analysis/walkforward.py` | Anchored and rolling walk-forward fold builders with strict ordering validation. |
| `analysis/bootstrap.py` | Politis-Romano stationary block bootstrap. Resamples daily-PnL blocks with autocorrelation preserved; re-applies the full TopStep rule book to each draw to estimate pass probability + Wilson CI + failure-mode breakdown. |
| `strategy/base.py` | `Strategy` protocol + `TargetPosition` dataclass. Strategies see a read-only `StrategyContext`. |

## What's NOT built yet

- **HMM regime filter** — `hmmlearn` already in `pyproject.toml` deps; module stub at `src/topstep50k/regime/` (empty).
- **Sensitivity analysis** — parameter grid sweep against the OOS test fold.
- **Rolling-window stats** — windowed Sharpe/DD over the equity curve.
- **Multi-asset portfolio layer** — ledger is multi-symbol-aware, but no portfolio coordinator yet.
- **Multi-asset correlation estimator** — for correlation-aware position sizing.
- **EV-gated risk sizing** — Kelly / fractional Kelly that only activates when the OOS edge is statistically positive.
- **Baseline strategy implementations** — no concrete strategy yet (Donchian, MR, etc.).
- **Real-data integration test** — blocked on data (see below).

## Data — current blocker

The three cleaned bar files (`es_cleaned.txt`, `nq_cleaned.txt`,
`gc_cleaned.txt`, ~110 MB each) live in
[this Google Drive folder](https://drive.google.com/drive/folders/1fVuZkbi8vhRwCSmro2CzUsnuM_jWwBjQ).

The sandbox network policy **blocks `drive.google.com`**
(`x-deny-reason: host_not_allowed`), so neither `gdown` nor `curl` can
reach Drive regardless of sharing settings. The MCP Drive
`download_file_content` tool returns the file as base64 in a single
message — at ~150 MB per file that exceeds any context window.

**Resolution path:** upload the three files as assets on a GitHub
Release (`https://github.com/biduchen9-afk/topstep-50k/releases/new`,
tag `data-v1`). GitHub release assets are reachable from the sandbox
(`objects.githubusercontent.com` is on the allow-list) and support
files up to 2 GB. Once uploaded, the next session can `curl` the
assets into `data/raw/` in seconds.

## How to pick up next session

1. Clone the repo and `pip install -e .[dev]`.
2. Run `pytest` — expect 57 passing.
3. If the GitHub release exists:
   - `curl -L -o data/raw/es_cleaned.txt <release asset URL>` (× 3 files).
   - `python -c "from topstep50k.data.loaders import load_bars_df; print(load_bars_df('data/raw/es_cleaned.txt').head())"` to verify the schema sniffer.
4. Pick the next chunk — recommended order:
   - **Baseline strategy + real-data smoke test** (Donchian breakout on ES). Smallest possible PR that exercises the whole stack on real bars.
   - **Walk-forward harness wiring**: split bars by fold, run engine per fold, aggregate OOS stats.
   - **HMM regime overlay**: train HMM on returns in the train fold only, predict regime on test fold, gate strategy by regime.
   - **Bootstrap on OOS daily-PnL**: feed walk-forward OOS series to `topstep_pass_probability()`.
   - **Sensitivity sweep**: parameter grid over the strategy, report pass-rate-vs-param surface.
   - **Multi-asset + correlation**: portfolio coordinator with per-symbol Ledger entries + rolling correlation gate.

## Recently relevant SSRN-type references

When wiring HMM and bootstrap layers, the methodology should follow:
- Politis & Romano (1994), "The Stationary Bootstrap" — already implemented in `bootstrap.py`.
- Hamilton (1989) — HMM regime switching, baseline.
- Bailey, López de Prado et al. on the Probabilistic Sharpe Ratio and
  Deflated Sharpe Ratio (SSRN id 1821643, 2460551). Stronger than raw
  Sharpe when reporting OOS skill on a single backtest.
- López de Prado, "The 7 Reasons Most Machine Learning Funds Fail"
  (SSRN id 3031282) — informs the walk-forward + purged-CV convention.

## Repo layout

```
src/topstep50k/
  rules/          # TopStep rule encoders
  engine/         # clock, types, ledger, backtest loop
  data/           # DataSource protocol + loaders
  audit/          # append-only event log
  analysis/       # stats, walk-forward, bootstrap
  strategy/       # Strategy protocol
  regime/         # (empty — HMM goes here)
  risk/           # (empty — EV-gated sizing goes here)
tests/
  unit/           # per-module
  integration/    # end-to-end backtest scenarios
docs/
  rules_sources.md
data/
  raw/            # gitignored; bar files land here
  processed/      # gitignored; parquet caches
```

## License

TBD.
