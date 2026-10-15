"""Paper-trading runner: consumes one-minute bars (replay or live), runs the sleeves bar by bar, steps the Topstep farm session by session, keeps a thread-safe snapshot for the dashboard
and persists its state after every completed session (so it can be stopped and resumed; missed sessions are replayed from the feed)."""
from __future__ import annotations

import math
import os
import pickle
import threading
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from paper.farm import DEFAULT_GROUPS, PaperFarm
from paper.feed import Bar, in_session
from paper.history import History
from paper.sleeves import CASH_OPEN, FLAT_BY, K_DEC, STEP, NoiseAreaSleeve, VwapAtrSleeve, OR_END

GROUP_SLEEVE = {"noise": "nq_noise", "gap": "nq_gap30"}
POINT_USD_PER_MICRO = 2.0           # MNQ: $2.00 per index point (NQ $20 / 10)
ROLL_REL_JUMP = 0.0008              # a one-minute splice of 8bp or more inside a roll window marks a contract roll (the back-test skips that session)


def third_friday(year: int, month: int) -> pd.Timestamp:
    first = pd.Timestamp(year=year, month=month, day=1)
    return first + pd.Timedelta(days=(4 - first.dayofweek) % 7 + 14)


def in_roll_window(label) -> bool:
    ts = pd.Timestamp(label)
    for m in (3, 6, 9, 12):
        tf = third_friday(ts.year, m)
        if tf - pd.Timedelta(days=10) <= ts <= tf:
            return True
    return False


class PaperRunner:
    def __init__(self, hist: History, farm: PaperFarm, cost_pts: float, state_path: Optional[str] = None, meta: Optional[dict] = None, R: float = 9600.0, t0: int = 0):
        self.hist, self.farm, self.cost = hist, farm, cost_pts
        self.R = R
        self.state_path = state_path
        self.meta = meta or {}
        self.lock = threading.RLock()
        self.t_next = t0                       # global session counter (the farm's `t`)
        self.last_label = None                 # label of the last COMPLETED session
        self.cur_label = None
        self.journal: List[dict] = []          # one row per completed session
        self.sleeves: Dict[str, object] = {}
        self.ctx: dict = {}
        self.reset_session_arrays()
        self.n_bars_total = 0
        self.last_bar_ts: Optional[pd.Timestamp] = None
        self.started_wall = time.time()

    # ------------------------------------------------------------------ session lifecycle
    def reset_session_arrays(self) -> None:
        self.bars: List[Bar] = []
        self.cpv = self.cv = 0.0
        self.vw_session: List[float] = []
        self.roll_today = False
        self.roll_note = ""
        self.zero_vol = 0
        self.bad_data = False

    def _make_sleeves(self) -> None:
        self.sleeves = {"noise": NoiseAreaSleeve(self.cost), "gap": VwapAtrSleeve(self.cost, 0.3, 0.2, OR_END, True)}

    def _start_session(self, label) -> None:
        self.cur_label = label
        self.reset_session_arrays()
        self._make_sleeves()
        c = self.hist.context()
        ok = math.isfinite(c["atr"]) and math.isfinite(c["prev_close"])
        c["ok"], c["roll"] = ok, False
        self.ctx = c
        atr_usd = c["atr"] * POINT_USD_PER_MICRO if ok else float("nan")
        self.atr_usd_today = atr_usd
        nd = max(1, int(round(self.R / atr_usd))) if ok else 1
        today = {g: dict(ok=ok, atr_usd=atr_usd, nd=nd) for g in GROUP_SLEEVE}
        for s in self.sleeves.values():
            s.reset(c)
        self.farm.begin_session(self.t_next, label, today)

    def _finish_session(self) -> None:
        for s in self.sleeves.values():
            s.finish()
        day, row = {}, dict(date=str(self.cur_label), t=self.t_next, roll=self.roll_today or self.bad_data, bars=len(self.bars), zero_vol=self.zero_vol, atr_usd=float(getattr(self, "atr_usd_today", float("nan"))))
        for g, s in self.sleeves.items():
            pts, low, tr = s.result()
            if self.roll_today or self.bad_data or not self.ctx.get("ok", False):
                pts = low = 0.0
            day[g] = dict(x=pts * POINT_USD_PER_MICRO, lo=low * POINT_USD_PER_MICRO)
            row[g] = dict(usd=day[g]["x"], low=day[g]["lo"], trades=tr)
        self.farm.end_session(day)
        if self.bars:
            o = np.array([b.o for b in self.bars]); h = np.array([b.h for b in self.bars]); l = np.array([b.l for b in self.bars]); c = np.array([b.c for b in self.bars]); sm = np.array([b.smin for b in self.bars])
            self.hist.add_bars(o, h, l, c, sm, self.roll_today)
        row["farm_net"] = self.farm.net
        row["cash"] = self.farm.cash
        self.journal.append(row)
        self.journal = self.journal[-5000:]
        self.last_label = self.cur_label
        self.cur_label = None
        self.t_next += 1
        self.save()

    # ------------------------------------------------------------------ bars
    def process(self, bar: Bar) -> None:
        if not in_session(bar):
            return
        with self.lock:
            lab = bar.label
            if self.last_label is not None and lab <= self.last_label:
                return                                     # already processed (resume overlap)
            if self.cur_label is not None and lab != self.cur_label:
                self._finish_session()
            if self.cur_label is None:
                self._start_session(lab)
            smin = bar.smin
            if self.bars and not self.roll_today and in_roll_window(lab) and smin < FLAT_BY:
                prev_c = self.bars[-1].c
                if prev_c and abs(bar.o - prev_c) / prev_c >= ROLL_REL_JUMP:
                    self.roll_today = True
                    self.roll_note = f"contract roll suspected at {bar.ts.strftime('%H:%M')} ET (jump {bar.o - prev_c:+.2f}): flat and no more trades today"
                    for s in self.sleeves.values():
                        s.finish()
            self.bars.append(bar)
            if bar.v <= 0:
                self.zero_vol += 1
            if not self.bad_data and len(self.bars) >= 30 and self.zero_vol / len(self.bars) > 0.5:
                self.bad_data = True
                self.roll_note = "feed carries no volume: the VWAP-based sleeves cannot be computed, session not traded"
                for sl in self.sleeves.values():
                    sl.finish()
            tp = (bar.h + bar.l + bar.c) / 3.0
            self.cpv += tp * bar.v
            self.cv += bar.v
            self.vw_session.append(self.cpv / max(self.cv, 1.0))
            if not self.roll_today and not self.bad_data:
                for s in self.sleeves.values():
                    s.on_bar(len(self.bars) - 1, smin, bar.o, bar.h, bar.l, bar.c, bar.v)
            self.n_bars_total += 1
            self.last_bar_ts = bar.ts

    def close_open_session(self) -> None:
        """Finish the session in progress (end of a replay or of the data)."""
        with self.lock:
            if self.cur_label is not None:
                self._finish_session()

    # ------------------------------------------------------------------ persistence
    def save(self) -> None:
        if not self.state_path:
            return
        tmp = self.state_path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(dict(hist=self.hist, farm=self.farm, t_next=self.t_next, last_label=self.last_label, journal=self.journal, cost=self.cost, meta=self.meta, R=self.R), f)
        os.replace(tmp, self.state_path)

    @classmethod
    def load(cls, path: str) -> "PaperRunner":
        with open(path, "rb") as f:
            d = pickle.load(f)
        r = cls(d["hist"], d["farm"], d["cost"], path, d["meta"], d["R"], d["t_next"])
        r.last_label, r.journal = d["last_label"], d["journal"]
        return r

    def next_label(self):
        """The first session label after the last completed one (weekdays only)."""
        if self.last_label is None:
            return None
        d = pd.Timestamp(self.last_label) + pd.Timedelta(days=1)
        while d.weekday() >= 5:
            d += pd.Timedelta(days=1)
        return d.date()

    # ------------------------------------------------------------------ dashboard snapshot
    def _sleeve_view(self, g: str, last_close: float) -> dict:
        s = self.sleeves.get(g)
        if s is None:
            return {}
        u = POINT_USD_PER_MICRO
        return dict(name=GROUP_SLEEVE[g], pos=int(s.pos), entry=None if s.pos == 0 else float(s.entry), unreal_usd=s.unrealized_pts(last_close) * u if s.pos else 0.0, realized_usd=s.realized * u,
                    day_low_usd=s.day_low_pts * u, trades=int(s.trades), stop=(float(getattr(s, "stop", float("nan"))) if s.pos and hasattr(s, "stop") else None), tradable=bool(s.tradable),
                    log=[dict(dir=int(t["dir"]), entry=float(t["entry"]), entry_smin=int(t["entry_smin"]), exit=(float(t["exit"]) if "exit" in t else None), exit_smin=(int(t["exit_smin"]) if "exit_smin" in t else None),
                              usd=(float(t["pnl_pts"]) * u if "pnl_pts" in t else None), reason=t.get("reason")) for t in (s.trade_log + ([s._cur] if s.pos else []))])

    def _accounts_view(self, last_close: float) -> List[dict]:
        u = POINT_USD_PER_MICRO
        out = []
        for g in self.farm.groups:
            s = self.sleeves.get(g.name)
            for er in g.evals:
                out.append(self._acct_row(g.name, "Combine", er.id, er.acct, er.k, s, last_close, u, er.acct.ph.target, None))
            for fr in g.funded:
                a = fr.acct
                trig = a.plan.t1 if a.stage == 1 else a.plan.t2
                row = self._acct_row(g.name, "Funded", fr.id, a, fr.k, s, last_close, u, trig, a)
                row.update(stage=a.stage, wins=a.wins, payouts=[round(p, 2) for p in a.pays], needs_wins=a.spec.win_days_req)
                out.append(row)
            for er, since in g.waiting:
                out.append(dict(group=g.name, kind="Passed - waiting for a funded slot", id=er.id, k=0, bal=0.0, live_bal=0.0, floor=0.0, cushion=0.0, target=0.0, today_usd=0.0, status="waiting"))
        return out

    def _acct_row(self, g, kind, id_, a, k, s, last_close, u, target, funded):
        day = (s.realized + s.unrealized_pts(last_close)) * u * k if s is not None and not self.roll_today else 0.0
        low = (s.day_low_pts * u) if s is not None and not self.roll_today else 0.0
        try:
            st = a.intraday(low, k) if k else "ok"
        except Exception:
            st = "ok"
        live = a.bal + (day if st == "ok" else (-3000.0 if st == "dll" else day))
        return dict(group=g, kind=kind, id=id_, k=int(k), bal=float(a.bal), live_bal=float(live), floor=float(a.floor), cushion=float(a.bal - a.floor), target=float(target), today_usd=float(day),
                    day_low_usd=float(low * k), status={"ok": "trading", "dll": "daily loss limit", "breach": "WOULD BREACH"}[st])

    def snapshot(self) -> dict:
        with self.lock:
            lastb = self.bars[-1] if self.bars else None
            lc = lastb.c if lastb else float("nan")
            sig = self.ctx.get("sigma", np.full(K_DEC, np.nan)) if self.ctx else np.full(K_DEC, np.nan)
            bands = []
            if self.ctx and self.ctx.get("ok") and self.bars:
                a0 = next((b for b in self.bars if b.smin >= CASH_OPEN), None)
                if a0 is not None:
                    hi_b, lo_b = max(a0.o, self.ctx["prev_close"]), min(a0.o, self.ctx["prev_close"])
                    bands = [dict(smin=CASH_OPEN + STEP * (k + 1), ub=hi_b * (1 + sig[k]), lb=lo_b * (1 - sig[k])) for k in range(K_DEC) if math.isfinite(sig[k])]
            chart = dict(smin=[b.smin for b in self.bars][::1], o=[b.o for b in self.bars], h=[b.h for b in self.bars], l=[b.l for b in self.bars], c=[b.c for b in self.bars], vwap=self.vw_session)
            sm = self.farm.summary()
            return dict(
                meta=self.meta, now=str(self.last_bar_ts) if self.last_bar_ts is not None else None, wall=time.time(), bars_total=self.n_bars_total,
                session=dict(label=str(self.cur_label) if self.cur_label else None, t=self.t_next, bars=len(self.bars), last=lc, last_smin=(lastb.smin if lastb else None), atr_pts=self.ctx.get("atr") if self.ctx else None,
                             prev_close=self.ctx.get("prev_close") if self.ctx else None, ok=bool(self.ctx.get("ok")) if self.ctx else False, roll=self.roll_today or self.bad_data, roll_note=self.roll_note,
                             flat_by_smin=FLAT_BY, last_completed=str(self.last_label) if self.last_label else None),
                sleeves={g: self._sleeve_view(g, lc) for g in GROUP_SLEEVE}, accounts=self._accounts_view(lc), farm=sm, bands=bands, chart=chart,
                journal=self.journal[-60:], net_curve=[[str(d), float(v)] for d, v in self.farm.net_curve[-400:]], events=self.farm.log[-60:],
                rules=dict(combine_target=9000, mll=4500, dll=-3000, consistency=0.55, first_payout_balance=self.farm.plan.t1, a1=self.farm.plan.a1, a2=self.farm.plan.a2, R=self.R, point_usd_per_micro=POINT_USD_PER_MICRO))
