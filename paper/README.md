# Paper-trading / forward-test harness

What it is: the frozen baseline (3 noise-area + 2 gap-VWAP Topstep accounts, balanced sizing) run bar by bar on a one-minute feed, with simulated Combine and funded accounts under the Topstep rules of `firm_engine`, a farm manager (one Combine at a time, fees, activations, payouts) and a live web page. Paper trading only: no orders are ever sent.

Parity (see `test_paper.py`): the event-driven sleeves reproduce the vectorised back-test on all 4,058 tradable sessions x 3 sleeves except one truncated-data session; the incremental accounts equal `run_eval_firm` / `run_funded_firm`; the farm equals `multi_firm.run_multifirm`.

## Run it (on your own machine: the dashboard is served from the machine that runs the harness)
```
pip install numpy pandas            # nothing else is needed
export TOPSTEP_RAW_DIR=/path/to/data_raw     # the stored Databento sessions (warm-up history); see the repo README
# 1. check that the vault has the instrument (the key is read from the environment, never stored)
LSE_API_KEY=... python -m paper.lse_probe NQ MNQ
# 2. download bars and look at the data-quality report (start in June to overlap the stored data)
LSE_API_KEY=... python -m paper.lse_fetch --symbol <SYMBOL> --start 2026-06-01 --out data_new/nq_lse_1m.pkl
# 3a. forward test over the bars (report: results/forward_test_report.json, rules in docs/PREREGISTRATION_15.md)
python -m paper.forward_test --bars data_new/nq_lse_1m.pkl
# 3b. live paper trading with the dashboard on http://127.0.0.1:8765/
LSE_API_KEY=... python -m paper.run --source lse --symbol <SYMBOL> --port 8765
# demo without any feed (replays stored data): python -m paper.run --source replay --from 2026-04-01 --speed 600
```
`--stop-at "2026-06-24 14:30"` leaves a replayed session open to look at a mid-session state; `--state` keeps a resumable pickle (missed sessions are fetched and replayed on restart).

## Notes
* The LSE adapter was written from the vendor SDK's public behaviour (REST `GET https://api.londonstrategicedge.com/vault/candles?symbol&timeframe=1m&start&end&limit&order`, header `x-api-key`); it is unit-tested against a fake server only. Whether the vault carries CME NQ/MNQ one-minute bars with volume is unknown until step 1; the vault's 'Futures' class is labelled `eurex` in the SDK.
* The harness skips a session when a contract-roll jump (>= 8 bp in one minute inside the quarterly roll window) is seen, and when most bars carry no volume.
* Time handling: bar timestamps are converted to America/New_York; sessions run 18:00 -> 16:00 ET and are labelled with the date of their 16:00 end.

## Running it live on your own machine (no stored history needed)  [added 15 October 2026]
The live dashboard is served by the machine that runs the harness, so run it on a computer of yours that stays on during the US session (18:00 to 16:00 ET, Sunday evening to Friday). Needs Python 3.10+, `pip install numpy pandas`, this repository, and the LSE key in the environment (never in a file):
```
export LSE_API_KEY=...            # PowerShell: $env:LSE_API_KEY="..."
python -m paper.run --source lse --symbol NQ.F --dataset futures --warm-from-lse 60 --port 8765
```
then open http://127.0.0.1:8765/ . `--warm-from-lse 60` downloads the last 60 days of one-minute candles, builds the ATR / noise-area history from them (sessions with holes or a suspected contract roll are kept out), and starts the farm fresh on the next session; the state is saved to `paper_state.pkl` after every session and a restart resumes it. Real-feed guards are on: a front-month splice at the session open inside a roll window (first bar 0.4% or more from the previous close) and sessions with holes in regular hours (fewer than 370 of 385 bars, early closes excepted).
**Known limits of the LSE vault candles (FINDINGS section 32):** 24% of the sessions after 2026-07-02 have holes in regular hours (two with multi-hour outages), and the September roll was messy. A live run on these candles is a machinery run, not a valid forward test (PR18 data check); the dashboard marks sessions with holes. `python -m paper.make_warmup` is the offline equivalent when the stored Databento history is available.
