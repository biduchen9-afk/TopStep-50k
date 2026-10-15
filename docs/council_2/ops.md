# Council 2, ops seat: data and forward-test operations (15 October 2026)

**Position.** The tick harness is not ready: six silent real-stream defects, two of which make a perfect feed read INVALID at the first PR18 look or rebuild sessions from hole-ridden vault candles after a weekend restart; a tested patch is appended. The forward test is a monitor: no rule change sharpens it and "not stopped" at 250 sessions is only a 2.3:1 signal, so run it for $0 as a veto, never as a gate. No clean independent sample exists in our data, and the Databento files live in one reclaimable container: back them up first.

## Evidence ([C] computed, [R] read; scripts: `scratchpad/council2/ops/`)
**Handled** [C]: DST, all timestamp forms, session boundaries, reconnect, a mid-session kill (journal identical after replay), a 300k-tick minute.
**Fail, unpatched** [C]:
1. 13:15 ET early closes (Black Friday, 24 Dec) count as holes (34 of 4,137 perfect Databento sessions), so PR18's 2% clause reads INVALID at sessions 49-99.
2. A session closes only when the next one's first bar arrives: Friday is unsaved all weekend, and a restart over 23 h later rebuilds it from vault candles (`run.py` [R]).
3. A silent socket never reconnects; auth errors are swallowed and retried every 5 s forever; after 49 h idle the reconnect start is stale.
4. A replay pause over 3 s cuts a minute (58 contracts became 14); the `replay` flag is unused.
5. One tick stamped +45 min blacks out 45 minutes; bad prints, NaN size, duplicates and a cumulative `volume` field (2,074x) corrupt bars unfiltered.
6. The roll guard covers only 8-18 Dec and flags a 0.10% news jump (LSE September: 6 of 74 sessions skipped [R]).

**Volume:** LSE candle volume equals Databento contracts in 99.2% of 27,179 June minutes, so ticks probably carry sizes. Tick counts move a year's pooled R +0.2, equal weights -1.06.
**Power** (20,000 bootstrap paths): stopped by 60/120/250: full edge 3/14/23%, zero edge 12/43/66%. LR of STOP 3.6/3.1/2.8; of "not stopped" 1.1/1.5/2.3. One look at 250 with the same 23% false-stop stops zero edge 70%. Confirming a full edge takes 993 sessions (3.9 years), half an edge 15. Combine passes (about 16 a year) have 16-18% power against 24%. Session 20 = 13 Nov, 60 = 12 Jan 2027, 120 = 7 Apr, 250 = 6 Oct.
**Independent sample:** ES-NQ daily RTH correlation 0.88 (dev), 0.94 (2023+); ES/GC ports were already tried (104 trials, FINDINGS 16-17); LSE 2016+ candles, if they exist, only cross-check Databento.

## Recommended next steps
1. **Today: back up Databento** (30 min). Commit only a sha256 manifest. `tar`+`xz` the three raw text files (about 150 MB) and the repaired NQ npz (220 MB), `age`-encrypt (passphrase in the user's password manager), split into 90 MB parts, push to a new PRIVATE repo, copy to the user's PC, restore once, compare hashes. Check the licence.
2. **Apply the patch, then pre-register Amendment A** (3 h): calendar early closes; INVALID does not latch, and two consecutive INVALID looks after 60 end the test as a data failure; **completeness gate at session 20**: two or more non-calendar hole sessions means the feed is unusable (97% caught at a 24% hole rate, 6% false alarm at 2%); Combine passes descriptive only; volume basis must be size or trade count; holes are never patched, only a full-session replay within 23 h repairs. Count from Sunday 18 Oct. Chance of a valid 120-session verdict about 35%, of changing a decision about 15%.
3. **Always-on, all $0; two runners should write identical journals.** Own computer: live, or `--max-idle 120 --no-server` once daily 16:00-17:00 ET (rebuilds the session from the 24 h replay); risks: sleep, key in shell history (use a 600-permission env file), one disk (copy state and `_bars.csv` nightly); a spare device adds SD corruption. GitHub Actions: daily cron only (6 h cap), PRIVATE repo, key as secret; public-repo schedules lapse after 60 idle days; ToS tie Actions to the repo's own lifecycle (gray; from memory). Claude scheduled task: reaches LSE only if the environment allows both LSE hosts (worked 15 Oct [R]; WebSocket via proxy untested), no disk, uses plan quota. Vercel, Supabase, free VMs: unsuitable or need a card (unverified).
4. **Ask LSE** if live ticks and replay are complete and share the vault's pipeline; compare live, replay and candle bar counts over 5 sessions.

**Go/no-go this week.** GO needs all: backup restored; patch applied, 15 tests pass; Day-0 sample shows size, UTC stamp, replay flag and 60 live minutes matching candles; amendment committed. NO-GO: no size field, quote ticks, replay under 22 h, over 5% mismatch on non-hole minutes.

**What not to do.** Re-tune looks; build an ES/GC "independent" sample (about 0% chance of new information); decide on Combine passes; use vault candles; put data, keys or state in the public repo.

**Biggest thing the chair may be wrong about.** That ticks escape the holes: the missing minutes were empty at 1-second resolution too (FINDINGS 33), so the loss is probably upstream; I give clean live ticks 40%.

**Questions for the user.** Which machine, awake 16:00-17:00 ET daily? May the key sit in a private GitHub secret or Claude environment? Is a card-on-file free VM acceptable? Does the LSE plan allow two streams? Will you re-run the one-call history probe? May an encrypted archive go to a private repo?

---
## Appendix: proposed patch (code, not counted in the 900 words)
Tested on a copy: 15/15 repo tests, the stress scripts, `patch -p1` clean. Full applicable diff including the new `paper/verdict.py` (offline weekly PR18 verdict from the journal and `results/forward_thresholds_R.json`, 0 mismatches against `kill_reasons` on 300 windows): `docs/council_2/ops_patch.diff`. Hunks of the three edited files:

```diff
--- paper/runner.py
@@ -58,0 +59 @@
+        self.bar_log: Optional[str] = None      # append every processed bar here (UTC ISO, o, h, l, c, v): the harness's own record of the live sample (restarts re-append a replayed session: keep the last copy per timestamp)
@@ -76 +77 @@
-        if len(r) and 1135 <= r.max() <= 1145 and len(r) >= 0.95 * (r.max() - 930 + 1):                    # early close: the session runs to about 13:00 ET (last bar 12:55 to 13:05) with no holes
+        if len(r) and 1135 <= r.max() <= 1160 and len(r) >= 0.95 * (r.max() - 930 + 1):                    # early close: 13:00 or 13:15 ET (last bar 12:55 to 13:20) with no holes
@@ -126,0 +128,3 @@
+            with self.lock:
+                if self.cur_label is not None and bar.smin >= 1320 and bar.label == self.cur_label:
+                    self._finish_session()                 # 16:00 ET reached: close the session now (farm, journal, state file), not at the next evening's first bar
@@ -135,0 +140,2 @@
+            if self.bars and bar.ts <= self.bars[-1].ts:
+                return                                     # duplicate or out-of-order bar (REST/tick seam, replay overlap)
@@ -149,0 +156,3 @@
+            if self.bar_log:
+                with open(self.bar_log, "a") as f:
+                    f.write(f"{bar.ts.tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%S')},{bar.o},{bar.h},{bar.l},{bar.c},{bar.v}\n")
--- paper/tickfeed.py
@@ -12,0 +13 @@
+import math
@@ -46 +47,2 @@
-                 clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep, max_reconnects: Optional[int] = None):
+                 clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep, max_reconnects: Optional[int] = None,
+                 stall: Optional[float] = None, max_replay_hours: Optional[float] = None, max_future: float = 90.0):
@@ -52,0 +55,10 @@
+        self.stall, self.max_replay_hours, self.max_future = stall, max_replay_hours, max_future
+        self.last_error: Optional[str] = None
+        self.n_future = self.n_bad_price = self.n_bad_size = self.n_stall = 0
+        self._fail_streak = 0; self._fut_run = 0
+
+    @staticmethod
+    def market_open(ts_utc: pd.Timestamp) -> bool:
+        """CME equity-index Globex hours in ET: Sunday 18:00 to Friday 17:00, daily halt 17:00-18:00 (holidays not modelled)."""
+        t = ts_utc.tz_convert(ET); m, wd = t.hour * 60 + t.minute, t.weekday()
+        return not (wd == 5 or (wd == 4 and m >= 1020) or (wd == 6 and m < 1080) or 1020 <= m < 1080)
@@ -56,0 +69,3 @@
+        if not (math.isfinite(price) and price > 0):
+            self.n_bad_price += 1
+            raise ValueError("bad price")
@@ -64,0 +80,2 @@
+        if size is not None and not (math.isfinite(size) and size >= 0):
+            self.n_bad_size += 1; size = 0.0                           # NaN / negative size: count the tick, not the contracts
@@ -85,0 +103 @@
+                    self.last_error = f"{type(e).__name__}: {e}"[:200]
@@ -88,0 +107,4 @@
+            if self.max_replay_hours is not None and src_start is not None:    # the vendor replays a limited window: never ask for an older start (a long outage becomes a hole, not a retry loop)
+                floor_ = (pd.Timestamp(self.clock(), unit="s", tz="UTC") - pd.Timedelta(hours=self.max_replay_hours)).ceil("min")
+                if pd.Timestamp(src_start, tz="UTC") < floor_:
+                    src_start = floor_.strftime("%Y-%m-%dT%H:%M:%S")
@@ -91 +113 @@
-            last_tick_wall = self.clock(); ended = False
+            last_tick_wall = self.clock(); ended = False; last_replay = False; kind = tk = None
@@ -97 +119,6 @@
-                    if cur is not None and now - last_tick_wall >= self.settle and now >= cur[0].timestamp() + 60 + self.settle:
+                    if self.stall and now - last_tick_wall > self.stall and self.market_open(pd.Timestamp(now, unit="s", tz="UTC")):
+                        self.n_stall += 1; self.last_error = f"no tick for {now - last_tick_wall:.0f} s while the market is open: reconnecting"
+                        ended = True
+                        break
+                    # wall-clock flush only for a live minute that ended minutes ago at most (never while ticks are being replayed: a replay pause must not cut a minute)
+                    if cur is not None and not last_replay and now - last_tick_wall >= self.settle and cur[0].timestamp() + 60 + self.settle <= now <= cur[0].timestamp() + 60 + 300:
@@ -108 +135 @@
-                self.n_ticks += 1
+                self.n_ticks += 1; self._fail_streak = 0
@@ -112,0 +140,8 @@
+                last_replay = replay
+                if not replay and minute > pd.Timestamp(self.clock(), unit="s", tz="UTC").floor("min") + pd.Timedelta(seconds=self.max_future):
+                    self.n_future += 1; self._fut_run += 1          # a tick stamped in the future would black out the real ones until its time comes
+                    if self._fut_run <= 30:
+                        continue                                    # one bad stamp is dropped; 30 in a row means the machine clock is slow: accept and say so
+                    self.last_error = "tick stamps run ahead of the machine clock: check the clock (NTP)"
+                else:
+                    self._fut_run = 0
@@ -127,0 +163,3 @@
+            if kind == "error":
+                self.last_error = f"{type(tk).__name__}: {tk}"[:200]
+            self._fail_streak += 1
@@ -136 +174 @@
-            self.sleep(self.backoff)
+            self.sleep(min(self.backoff * 2 ** min(self._fail_streak - 1, 6), 300.0))      # exponential, capped at 5 minutes
--- paper/run.py
@@ -49,0 +50 @@
+    ap.add_argument("--max-idle", type=float, default=None, help="ticks: batch mode, end the feed after this many seconds without a tick (run once a day between 16:00 and 17:00 ET with --no-server; the session is rebuilt from the vendor's 24 h replay)")
@@ -56,0 +58 @@
+    redact = lambda s: str(s).replace(os.environ.get("LSE_API_KEY") or "\0", "***")           # exception text can carry the URL; the snapshot (and the public viewer) must never carry the key
@@ -93,0 +96,2 @@
+    runner.bar_log = os.path.splitext(a.state)[0] + "_bars.csv"
+    tb_ref = None
@@ -111 +115 @@
-            tick_from = max(first, now - pd.Timedelta(hours=23))                 # the stream replays at most 24 hours; anything older is caught up from the vault candles
+            tick_from = max(first, (now - pd.Timedelta(hours=23)).ceil("min"))   # the stream replays at most 24 hours; anything older is caught up from the vault candles (whole minutes, so no partial first bar)
@@ -113 +117 @@
-            tb = TickBars(lambda s: sdk_ticks(a.symbol, key, s), start=tick_from.strftime("%Y-%m-%dT%H:%M:%S"))
+            tb = TickBars(lambda s: sdk_ticks(a.symbol, key, s), start=tick_from.strftime("%Y-%m-%dT%H:%M:%S"), stall=90.0, max_replay_hours=23.0, max_idle=a.max_idle)
@@ -114,0 +119 @@
+            tb_ref = tb
@@ -128,0 +134,2 @@
+                if tb_ref is not None:
+                    runner.meta.update(volume_basis=tb_ref.volume_basis, feed_reconnects=tb_ref.reconnects, feed_last_error=redact(tb_ref.last_error or ""), feed_rejected=dict(future=tb_ref.n_future, price=tb_ref.n_bad_price, size=tb_ref.n_bad_size, stall=tb_ref.n_stall))
@@ -137 +144 @@
-            runner.meta["error"] = f"{type(e).__name__}: {e}"
+            runner.meta["error"] = redact(f"{type(e).__name__}: {e}")
```
