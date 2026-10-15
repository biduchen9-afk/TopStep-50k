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

## Running it live on your own machine (no stored history needed)  [updated 15 October 2026]
The dashboard is served by the machine that runs the harness (a Claude cloud session cannot host a process you can open, and is reclaimed when idle), so run it on a computer of yours that stays on during the US session (18:00 to 16:00 ET, Sunday evening to Friday). Python 3.10+, `pip install numpy pandas lse-data`, this repository, and the LSE key in the environment (never in a file):
```
export LSE_API_KEY=...            # PowerShell: $env:LSE_API_KEY="..."
python -m paper.run --source ticks --symbol NQ.F --dataset futures --warm-from-lse 60 --port 8765
```
then open http://127.0.0.1:8765/ . `--source ticks` builds the one-minute bars from the LSE live tick stream (`paper/tickfeed.py`: replays up to 23 hours of missed ticks after a restart, older gaps are caught up from the vault candles; a dropped connection reconnects and rebuilds the unfinished minute, so nothing is counted twice), which avoids the holes in the vault's candles. `--warm-from-lse 60` builds the ATR / noise-area history from the last 60 days of candles (sessions with holes or a suspected roll are kept out) and starts the farm fresh on the next session. The state is saved to `paper_state.pkl` after every session and a restart resumes it. `--source lse` is the REST-candle fallback.
Real-feed guards are on: a front-month splice at a session open inside a roll window (first bar 0.4% or more from the previous close) and sessions with holes in regular hours (fewer than 370 of 385 bars, early closes excepted) are kept out of trading / the ATR history; the dashboard shows a chip with the number of sessions with holes.
Not verified against the real stream (the build container cannot reach LSE): the tick field names. `TickBars` takes the volume from the first of `size`, `volume`, `qty`, `quantity`, `last_size`, `amount`; if none exists it counts ticks instead (the dashboard metadata says which) and the VWAP-based sleeves would be tick-weighted, not volume-weighted: check one sample tick first (`python - <<< "from lse import LSE; import os; [print(vars(t)) for _, t in zip(range(5), LSE(api_key=os.environ['LSE_API_KEY']).stream(['NQ.F']))]"`).
Known limits of the vault candles (FINDINGS section 32): 24% of the sessions after 2026-07-02 have holes in regular hours (two with multi-hour outages), and the September roll was messy; a run on candles is a machinery run, not a valid forward test (PR18 data check). `python -m paper.make_warmup` is the offline equivalent when the stored Databento history is available.


## Optional: publish to the remote viewer (Vercel + Supabase)
Set `PAPER_PUBLISH_URL` (https://<project>.vercel.app/api/ingest) and `PAPER_PUBLISH_TOKEN` in the environment of the harness and it pushes its snapshot every `--publish-every` seconds (default 15) so the viewer (`api/` + `public/` at the repo root, notes in `web/README.md`) can be opened from any device. Setup, security model and limits: `web/README.md`. The engine always runs in the harness; the viewer only displays.
