# Remote viewer (Vercel + Supabase)

A read-only page you can open from any device. **The engine does not run here**: the Python harness (`python -m paper.run`) computes everything and pushes its snapshot to this viewer; Vercel only displays it. That is also why the strategy and model are guaranteed to be the ones tested in this project: it is the same code, the same costs, the same accounts.

```
harness (your computer, or any always-on machine)  --POST /api/ingest (bearer ingest token)-->  Vercel function  --rpc paper_put-->  Supabase (one row)
browser (this page, access key in localStorage)    --GET  /api/state  (x-dash-key)-------------> Vercel function  --rpc paper_get-->  Supabase
```

## What it can and cannot do
* **Can:** show the last published state 24/7, mark itself `OFFLINE` when the harness has not published for 90 s, work from a phone.
* **Cannot:** collect live data by itself. A Vercel function lives for seconds to minutes; the LSE tick stream is a WebSocket that must stay open all week. Something has to keep `python -m paper.run --source ticks ...` alive (your computer left on, a Raspberry Pi, a small always-on machine). When it is off, the page shows the last state and says `OFFLINE`.
* A cron on the free (Hobby) Vercel plan can run at most once a day, so a Vercel-only design could update the farm once per evening from the vault candles (which have holes since July), not live. Not built.

## Security model (nothing sensitive is stored on Vercel)
* The database (`web/supabase/schema.sql`) keeps the snapshot and two token **hashes** in a schema the REST API does not expose, with row-level security on and no policies. The only things the public key can reach are `paper_get(token)` and `paper_put(token, payload)`; both raise `forbidden` unless SHA-256(token) matches the stored hash. Tested as the `anon` role: wrong or null tokens are refused, the private tables cannot be read or written, the read token cannot publish and the ingest token cannot read.
* The two tokens are random 32-byte strings. Vercel holds only `SUPABASE_URL` and the **publishable** key (meant to be public). The read key lives in the browser you typed it into; the ingest token only on the harness machine (`PAPER_PUBLISH_TOKEN`, an environment variable, never a command-line argument and never part of the published data).
* No LSE key, no Databento data and no raw ticks reach Vercel or Supabase. The snapshot holds derived paper-trading state (account balances, the day's bars for the chart, the trade log).
* Pages send `noindex`, `no-store`, a strict content-security policy and `no-referrer`.
* Rotate a token: generate a new random one, put the SHA-256 hex of it in `paper_private.tokens` (statement at the bottom of `schema.sql`), use the new token.

## Deploy (about 3 minutes, free plan)
1. Vercel -> Add New -> Project -> import the GitHub repo `biduchen9-afk/TopStep-50k`. **Root Directory: `web`**. Framework preset: Other. Leave build and install commands empty.
2. Environment variables (Production and Preview): `SUPABASE_URL` = `https://urnxldnectikeekkjqzm.supabase.co`, `SUPABASE_KEY` = the project's publishable key (Supabase dashboard -> Project Settings -> API Keys -> `sb_publishable_...`).
3. The code lives on the branch `claude/sharpe-strategy-validation-kks7q1`, not on `main`: either merge it, or in the Vercel project set Settings -> Environments -> Production -> Branch Tracking to that branch, then redeploy.
4. Open the project URL, type the access key.
5. On the machine that runs the harness:
```
export PAPER_PUBLISH_URL=https://<your-project>.vercel.app/api/ingest
export PAPER_PUBLISH_TOKEN=<ingest token>
python -m paper.run --source ticks --symbol NQ.F --dataset futures --warm-from-lse 60 --port 8765
```
The local dashboard on port 8765 keeps working; the same snapshot is also pushed every 15 s (`--publish-every`).

## Free-plan limits to keep in mind
The page polls every 10 s while the tab is visible (one function call each); left open all week that is about 60,000 calls, far below the free allowance. The harness writes one small row every 15 s (about 5,000 a day); the row is overwritten, so the database does not grow.

## Files
`public/index.html` is a copy of `paper/static/dashboard.html` (a test fails if they differ; edit the source and copy it over), `api/state.js`, `api/ingest.js`, `api/_supabase.js`, `vercel.json`, `supabase/schema.sql`, `test_api.js` (`node --test web/test_api.js`).
