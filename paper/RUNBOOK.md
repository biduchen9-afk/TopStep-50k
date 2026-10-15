# Runbook: starting the forward test (PR19)

What you are starting: the paper farm (3 noise-area + 2 gap-VWAP accounts, one Combine at a time) fed by the live LSE tick stream, judged by `docs/PREREGISTRATION_19.md`. It places no real orders and costs nothing. The machine must stay on and awake: every minute it is off is a hole in the data, and the test ends as DATA FAILURE if the feed has holes (two hole sessions in the first 20 sessions, or more than 2% later). Expect to leave it running for years; a KILL, if it comes, is most likely after month 12.

## 0. One-time setup (any OS)
1. Python 3.10 or newer. `git clone https://github.com/biduchen9-afk/TopStep-50k.git`, `cd TopStep-50k`, `git checkout claude/sharpe-strategy-validation-kks7q1` (or `main` once merged).
2. `pip install numpy pandas lse-data`
3. **The LSE key.** The key that was pasted in the chat must be treated as exposed: in your LSE account, create a new key and revoke the old one. Never paste the new key into chat, a commit, an issue or a screenshot. Save it as the only line of a file outside the repository:
   * macOS / Linux: `printf '%s\n' 'PASTE_KEY_HERE' > ~/lse.key && chmod 600 ~/lse.key` (then clear the shell history line, or type a space before the command if your shell ignores such lines).
   * Windows (PowerShell): create `C:\Users\YOU\lse.key` in Notepad with the key as the only line, then right-click, Properties, Security, and leave only your own account.
   The harness reads it with `--key-file`; it is never printed or logged.

## 1. Day-0 check (about 70 minutes, during Globex hours: Sunday 18:00 ET to Friday 17:00 ET, not 17:00-18:00)
`python -m paper.day0 --symbol NQ.F --dataset futures --key-file ~/lse.key`
It shows which fields a tick carries, builds one-minute bars from the live ticks for an hour, compares them with the vendor's own candles and prints **GO** or **NO-GO** (and writes `day0_report.json`, which holds counts only). Rules from PR19: a size/volume field must exist; timestamps must be UTC with a replay flag; the tick-built close must be within one tick of the vendor candle in at least 95% of at least 60 minutes. **NO-GO means do not start the clock**: send the report back and we investigate (a feed without a size field cannot run the VWAP sleeves faithfully). A different tick field name than we guessed (`size`, `volume`, `qty`, `quantity`, `last_size`, `amount`) is the most likely first problem and is a small fix.

## 2. Start the clock (before an 18:00 ET open; intended Sunday 18 October 2026, for the Monday session)
Keep the machine awake first, then start the supervised harness. It restarts the harness if it exits, resumes the same journal after a restart (it never restarts the clock), logs to `paper_harness.log` and keeps rolling backups in `paper_backups/` (copy them elsewhere now and then; never into the public repository).
* **macOS:** `caffeinate -i python -m paper.supervise -- --source ticks --symbol NQ.F --dataset futures --warm-from-lse 60 --fresh --key-file ~/lse.key --port 8765` (also System Settings, Battery or Energy: prevent automatic sleeping on power adapter).
* **Windows:** Settings, System, Power: Sleep = Never (plugged in). In PowerShell: `python -m paper.supervise -- --source ticks --symbol NQ.F --dataset futures --warm-from-lse 60 --fresh --key-file C:\Users\YOU\lse.key --port 8765`. To start at logon, Task Scheduler, create a task "At log on" running the same command in the repository folder (drop `--fresh` in the task, so a reboot resumes the journal).
* **Linux:** `python -m paper.supervise -- --source ticks ... --fresh --key-file ~/lse.key --port 8765`. To start at boot, a systemd user unit with `ExecStart=/usr/bin/python3 -m paper.supervise -- --source ticks --symbol NQ.F --dataset futures --warm-from-lse 60 --key-file %h/lse.key --port 8765`, `WorkingDirectory=` the repository, `Restart=always` (no `--fresh` in the unit), and `loginctl enable-linger $USER`.
`--fresh` is used **once**, at day 0. Using it again deletes the journal and restarts the clock.

## 3. Looking at it
* Dashboard: http://127.0.0.1:8765/ on that machine (the PR19 chip shows the status, session count out of 756, pooled R and the kill line in its tooltip).
* Verdict on demand: `python -m paper.verdict --state paper_state.pkl` (status, pooled R, the shadow VWAP and basket columns, hole sessions, next look).
* Statuses: TOO EARLY (before session 63) / CONTINUE / WATCH (below the PR18 development 5th percentile: look, do not act) / KILL (cumulative pooled R crossed the line: stop and investigate) / INVALID (too many hole sessions at the latest look) / DATA FAILURE (feed unusable) / CONFIRMED or CANNOT TELL at session 756.

## 4. When something goes wrong
* The log shows repeated restarts: read the last lines of `paper_harness.log` (the key is redacted); a stream that is silent for 90 s during Globex hours reconnects by itself.
* The machine slept or lost power: restart the supervised command (without `--fresh`); missed minutes become hole sessions. Two hole sessions among the first 20 end the test, so protect the first four weeks.
* Never edit code, settings or the state file during the test; any change starts a new registration. Updating the repository and restarting is fine only if the change touches nothing under `paper/` (ask first).

## 5. If the machine cannot stay on (weaker option)
Batch mode, once a day between 16:00 and 17:00 ET: `python -m paper.run --source ticks --symbol NQ.F --dataset futures --key-file ~/lse.key --max-idle 120 --no-server` rebuilds the session from the vendor's 24-hour replay. Completeness is lower (a session is only as good as the replay), the first 20-session gate applies equally, and Friday's session must be run before Saturday 16:00 ET. Use it only until a machine that stays on is available.
