"""Incremental (day-by-day) Topstep account simulators. They apply exactly the per-day logic of firm_engine.run_eval_firm / run_funded_firm (parity-tested in test_paper.py) but keep
their state between days so that a live / paper-trading run can step them one session at a time and inspect them intraday.

Units: `x` = the day's net P&L per ONE micro in dollars, `lo` = the day's worst mark-to-market per micro (<= 0), `k` = contracts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from firm_engine import EvalPhase, FundedSpec, _floor_after, _scaled_a
from real_engine import FundedPlan, xfa_micro_cap


class EvalAccount:
    """A Combine (single phase): target, trailing maximum loss limit (real-time incl. open P&L), soft daily loss limit, consistency, micro cap."""
    kind = "combine"

    def __init__(self, phase: EvalPhase):
        self.ph = phase
        self.bal = self.peak = self.best = 0.0
        self.floor = -phase.dd
        self.days = 0              # days with a non-zero result
        self.age = 0               # sessions traded so far
        self.status: Optional[int] = None     # None running, 1 passed, 0 breached, -1 timed out
        self.today_k = 0

    def size(self, nd: int) -> int:
        return min(int(nd), self.ph.max_micros)

    # --- intraday view: would the account breach / hit the DLL if the day's worst MTM per micro is `lo` with k contracts?
    def intraday(self, lo: float, k: int) -> str:
        ph = self.ph
        w = lo * k
        lim = self.floor - self.bal
        first = lim if ph.dll is None else max(lim, ph.dll)
        if ph.dd_type == "intraday" and w <= first:
            return "dll" if (ph.dll is not None and ph.dll > lim and not ph.dll_hard) else "breach"
        if ph.dll is not None and w <= ph.dll:
            return "breach" if ph.dll_hard else "dll"
        return "ok"

    def end_day(self, x: float, lo: float, k: int) -> Optional[int]:
        ph = self.ph
        p, w = x * k, lo * k
        lim = self.floor - self.bal
        self.age += 1
        self.today_k = k
        if ph.dd_type == "intraday" or ph.dll is not None:
            first = lim if ph.dll is None else max(lim, ph.dll)
            if ph.dd_type == "intraday" and w <= first:
                if ph.dll is not None and ph.dll > lim and not ph.dll_hard:
                    p = ph.dll - ph.dll_slip
                else:
                    self.status = 0
                    return self.status
            elif ph.dll is not None and w <= ph.dll:
                if ph.dll_hard:
                    self.status = 0
                    return self.status
                p = ph.dll - ph.dll_slip
        if ph.day_cap is not None and p > ph.day_cap:
            p = ph.day_cap
        self.bal += p
        if ph.dd_type == "eod" and self.bal <= self.floor:
            self.status = 0
            return self.status
        if p != 0.0:
            self.days += 1
        if p > self.best:
            self.best = p
        if self.bal > self.peak:
            self.peak = self.bal
            self.floor = _floor_after(self.bal, self.peak, ph.trail, ph.dd, ph.lock, ph.lock_level, self.floor)
        if self.bal >= ph.target and self.days >= ph.min_days and (ph.consistency is None or self.best <= ph.consistency * self.bal):
            self.status = 1
        elif self.age >= ph.max_days:
            self.status = -1
        return self.status

    @property
    def cushion(self) -> float:
        return self.bal - self.floor

    def snapshot(self) -> dict:
        return dict(kind=self.kind, bal=self.bal, floor=self.floor, peak=self.peak, target=self.ph.target, age=self.age, days=self.days, best_day=self.best, status=self.status)


class FundedAccount:
    """An Express Funded Account: trailing loss limit that locks at the starting balance, scaling-plan micro cap, payout rules, floor reset after each payout."""
    kind = "funded"

    def __init__(self, spec: FundedSpec, plan: FundedPlan, t_start: int):
        self.spec, self.plan = spec, plan
        self.t_start = t_start
        self.bal = self.peak = 0.0
        self.floor = -spec.dd
        self.frozen = False
        self.stage = 1
        self.wins = self.tdays = 0
        self.prof = self.best = 0.0
        self.last_pay_t = -10 ** 9
        self.pays: List[float] = []
        self.pay_times: List[int] = []
        self.hist: List[float] = []
        self.hist_ps = 0
        self.age = 0
        self.state = 0            # 0 running, 1 dead, 2 closed by plan
        self.today_k = 0

    def size(self, atr_usd: float, nd: int, t: int, atr2_usd: Optional[float] = None) -> int:
        spec, plan = self.spec, self.plan
        if plan.a1 is None:
            mult = plan.r1 if self.stage == 1 else plan.r2
            k = max(1, int(round(nd * mult)))
        else:
            a_t = atr_usd if (self.stage == 1 or atr2_usd is None) else atr2_usd
            a = plan.a1 if (self.stage == 1 or plan.a2 is None) else plan.a2
            if self.stage == 1 and plan.scale is not None:
                a = _scaled_a(plan.scale, plan.a1, self.bal, t - self.t_start, self.hist)
            elif self.stage == 2 and plan.scale2 is not None:
                a = _scaled_a(plan.scale2, a, self.bal, t - self.last_pay_t - 1, self.hist[self.hist_ps:], len(self.pays))
            k = 1 if not (a_t > 0) else max(1, int(round(a * max(self.bal - self.floor, 250.0) / a_t)))
        return min(k, xfa_micro_cap(self.bal, spec.ladder))

    def intraday(self, lo: float, k: int) -> str:
        spec = self.spec
        w = lo * k
        lim = self.floor - self.bal
        if spec.dd_type == "intraday" or spec.dll is not None:
            first = lim if spec.dll is None else max(lim, spec.dll)
            if spec.dd_type == "intraday" and w <= first:
                return "dll" if (spec.dll is not None and spec.dll > lim and not spec.dll_hard) else "breach"
            if spec.dll is not None and w <= spec.dll:
                return "breach" if spec.dll_hard else "dll"
        return "ok"

    def end_day(self, x: float, lo: float, k: int, t: int) -> List[float]:
        """Apply one session; returns the payouts (after the split) made at the end of it. self.state: 1 dead, 2 closed."""
        spec, plan = self.spec, self.plan
        self.age += 1
        self.today_k = k
        p, w = x * k, lo * k
        lim = self.floor - self.bal
        if spec.dd_type == "intraday" or spec.dll is not None:
            first = lim if spec.dll is None else max(lim, spec.dll)
            if spec.dd_type == "intraday" and w <= first:
                if spec.dll is not None and spec.dll > lim and not spec.dll_hard:
                    p = spec.dll - spec.dll_slip
                else:
                    self.state = 1
                    return []
            elif spec.dll is not None and w <= spec.dll:
                if spec.dll_hard:
                    self.state = 1
                    return []
                p = spec.dll - spec.dll_slip
        if spec.day_cap is not None and p > spec.day_cap:
            p = spec.day_cap
        self.bal += p
        self.hist.append(p)
        if spec.dd_type == "eod" and self.bal <= self.floor:
            self.state = 1
            return []
        if p != 0.0:
            self.tdays += 1
        self.prof += p
        if p >= spec.win_day_amt:
            self.wins += 1
        if p > self.best:
            self.best = p
        if not self.frozen and self.bal > self.peak:
            self.peak = self.bal
            self.floor = _floor_after(self.bal, self.peak, spec.trail, spec.dd, spec.lock, spec.lock_level, self.floor)
        eligible = (spec.win_days_req == 0 or self.wins >= spec.win_days_req) and self.tdays >= spec.min_days_req and (t - self.last_pay_t) >= spec.gap_sessions
        if spec.consistency is not None:
            eligible = eligible and self.prof > 0 and self.best <= spec.consistency * self.prof
        if spec.min_cycle_profit > 0:
            eligible = eligible and self.prof >= spec.min_cycle_profit
        if spec.net_pos_after_first and self.pays:
            eligible = eligible and self.prof > 0
        trig, frac = (plan.t1, plan.w1) if self.stage == 1 else (plan.t2, plan.w2)
        out: List[float] = []
        if eligible and self.bal >= trig:
            gross = min(min(frac, spec.payout_frac) * self.bal, spec.payout_cap, self.bal - spec.buffer)
            if gross >= spec.payout_min:
                self.bal -= gross
                if spec.after_payout == "zero":
                    self.floor, self.frozen = spec.post_floor, True
                elif spec.after_payout == "reset_peak":
                    self.peak = self.bal
                    self.floor = self.bal - spec.dd
                self.wins = self.tdays = 0
                self.prof = self.best = 0.0
                self.last_pay_t = t
                self.hist_ps = len(self.hist)
                self.pays.append(spec.split * gross)
                self.pay_times.append(t)
                out.append(spec.split * gross)
                self.stage = 2
                if plan.t2 is None or len(self.pays) >= plan.max_payouts:
                    self.state = 2
        return out

    @property
    def cushion(self) -> float:
        return self.bal - self.floor

    def snapshot(self) -> dict:
        return dict(kind=self.kind, bal=self.bal, floor=self.floor, peak=self.peak, stage=self.stage, wins=self.wins, tdays=self.tdays, age=self.age, payouts=list(self.pays), state=self.state,
                    next_trigger=(self.plan.t1 if self.stage == 1 else self.plan.t2))
