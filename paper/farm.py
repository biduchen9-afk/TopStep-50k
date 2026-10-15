"""Incremental farm manager for the Topstep default layout: 3 noise-area + 2 gap-VWAP funded slots, ONE Combine open at a time (the next one is bought for the group with the most empty
slots, ties by list order), Combines billed every 21 sessions, a passed Combine waits at most 21 sessions for a free slot, funded accounts traded with the 'balanced' plan, payouts
arrive two sessions after they are made. It mirrors multi_firm.run_multifirm(Policy(reserve_cycles=None, global_one_eval=True)) session by session (parity-tested in test_paper.py)
but keeps its state between sessions, so a paper-trading run can step it live. Dollars are per ONE micro for the strategy inputs (x, lo, atr_usd) as in the back-test streams."""
from __future__ import annotations

import heapq
import math
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from firm_engine import FirmSpec, topstep_150k
from paper.accounts import EvalAccount, FundedAccount
from real_engine import FundedPlan

DEFAULT_GROUPS: Tuple[Tuple[str, int], ...] = (("noise", 3), ("gap", 2))
BALANCED_PLAN = FundedPlan("p", "x", 2000.0, 0.5, 1.0, 0.0, 0.5, 1.0, 99, 1.2, 0.8)


@dataclass
class EvalRec:
    id: int
    group: str
    acct: EvalAccount
    t_start: int
    k: int = 0
    bills: int = 1


@dataclass
class FundRec:
    id: int
    group: str
    acct: FundedAccount
    t_act: int
    k: int = 0
    n_paid_events: int = 0
    t_buy: int = -1                 # session on which the Combine of this account was bought
    first_pay_t: Optional[int] = None


@dataclass
class GroupState:
    name: str
    slots: int
    evals: List[EvalRec] = field(default_factory=list)
    funded: List[FundRec] = field(default_factory=list)
    waiting: List[Tuple[EvalRec, int]] = field(default_factory=list)
    n_evals: int = 0
    n_passed: int = 0
    n_funded: int = 0
    n_payouts: int = 0
    n_paid_accts: int = 0
    died: int = 0
    wasted: int = 0
    fees: float = 0.0
    payouts: float = 0.0
    first_buy_t: Optional[int] = None
    first_payout_t: Optional[int] = None


class PaperFarm:
    def __init__(self, groups: Sequence[Tuple[str, int]] = DEFAULT_GROUPS, spec: Optional[FirmSpec] = None, R: float = 9600.0, plan: FundedPlan = BALANCED_PLAN,
                 payout_delay: int = 2, max_wait: int = 21, p_est: float = 1.0, stop_new_at: Optional[int] = None, plans: Optional[Dict[str, FundedPlan]] = None):
        self.spec = spec or topstep_150k()
        self.R, self.plan = R, plan
        self.plans = plans or {}               # optional per-group funded plan (default: `plan` for every group)
        self.retired: List[FundRec] = []       # funded accounts that have ended (kept for statistics)
        self.payout_delay, self.max_wait, self.p_est = payout_delay, max_wait, p_est
        self.stop_new_at = stop_new_at          # parity mode: no purchases / activations from this session on (the batch engine stops at n_end - 5)
        self.groups = [GroupState(n, s) for n, s in groups]
        self.cash = 0.0
        self.cash_events: List[Tuple[int, int, float]] = []
        self._seq = 0
        self.t = -1
        self.date = None
        self.fees = 0.0
        self.payouts = 0.0
        self.next_id = 1
        self.log: List[dict] = []
        self.today: Dict[str, dict] = {}
        self.net_curve: List[Tuple[object, float]] = []

    # ------------------------------------------------------------------ helpers
    def _log(self, kind: str, group: str, **kw) -> None:
        self.log.append(dict(t=self.t, date=str(self.date), kind=kind, group=group, **kw))

    def group(self, name: str) -> GroupState:
        return next(g for g in self.groups if g.name == name)

    def _push_cash(self, t_arrive: int, amt: float) -> None:
        self._seq += 1
        heapq.heappush(self.cash_events, (t_arrive, self._seq, amt))

    # ------------------------------------------------------------------ session start
    def begin_session(self, t: int, date, today: Dict[str, dict]) -> None:
        """today[group] = dict(ok, atr_usd (per micro), nd (desired Combine micros)). Handles everything that is decided before the market opens."""
        spec = self.spec
        self.t, self.date, self.today = t, date, today
        while self.cash_events and self.cash_events[0][0] <= t:
            _, _, amt = heapq.heappop(self.cash_events)
            self.cash += amt
        allow_new = self.stop_new_at is None or t < self.stop_new_at
        # funded accounts that ended last session leave their slot; Combines that ended are resolved; waiting passes are activated
        for g in self.groups:
            alive = []
            for fr in g.funded:
                if fr.acct.state != 0:
                    self.retired.append(fr)
                    self._log("funded_died" if fr.acct.state == 1 else "funded_closed", g.name, id=fr.id, payouts=len(fr.acct.pays))
                else:
                    alive.append(fr)
            g.funded = alive
            if allow_new:
                keep = []
                for er in g.evals:
                    if er.acct.status is None:
                        keep.append(er)
                    elif er.acct.status == 1:
                        g.n_passed += 1
                        g.waiting.append((er, t))
                        self._log("eval_pass", g.name, id=er.id, sessions=er.acct.age)
                    else:
                        self._log("eval_fail" if er.acct.status == 0 else "eval_timeout", g.name, id=er.id, sessions=er.acct.age)
                g.evals = keep
                still = []
                for er, since in g.waiting:
                    if len(g.funded) < g.slots:
                        fee = spec.activation_fee
                        g.fees += fee; self.fees += fee; self.cash -= fee
                        g.n_funded += 1
                        fr = FundRec(er.id, g.name, FundedAccount(spec.funded, self.plans.get(g.name, self.plan), t), t, t_buy=er.t_start)
                        g.funded.append(fr)
                        self._log("activation", g.name, id=er.id, fee=fee)
                    elif t - since > self.max_wait:
                        g.wasted += 1
                        self._log("pass_wasted", g.name, id=er.id)
                    else:
                        still.append((er, since))
                g.waiting = still
        # purchase: one Combine open at a time across all groups, bought for the group with most empty funded slots (ties by order)
        if allow_new and not any(g.evals for g in self.groups):
            vac = [(g.slots - (len(g.funded) + len(g.waiting)), -gi) for gi, g in enumerate(self.groups)]
            best = max(vac)
            if best[0] > 0:
                g = self.groups[-best[1]]
                occupied = len(g.funded) + len(g.waiting)
                if len(g.evals) < spec.max_evals and occupied + self.p_est * len(g.evals) < g.slots:
                    price = spec.eval_fee
                    er = EvalRec(self.next_id, g.name, EvalAccount(spec.eval_phases[0]), t)
                    self.next_id += 1
                    g.evals.append(er)
                    g.n_evals += 1
                    g.fees += price; self.fees += price; self.cash -= price
                    if g.first_buy_t is None:
                        g.first_buy_t = t
                    self._log("eval_fee", g.name, id=er.id, fee=price)
        # billing of Combines that are still open at the start of their 22nd, 43rd, ... session
        for g in self.groups:
            for er in g.evals:
                if er.acct.age > 0 and er.acct.age % spec.cycle == 0 and er.acct.status is None:
                    g.fees += spec.eval_fee; self.fees += spec.eval_fee; self.cash -= spec.eval_fee
                    er.bills += 1
                    self._log("eval_fee", g.name, id=er.id, fee=spec.eval_fee)
        # contracts for today
        for g in self.groups:
            td = today[g.name]
            nd = int(td.get("nd", 1))
            for er in g.evals:
                er.k = er.acct.size(nd)
            for fr in g.funded:
                fr.k = fr.acct.size(td.get("atr_usd", float("nan")), nd, t)

    # ------------------------------------------------------------------ session end
    def end_session(self, day: Dict[str, dict]) -> None:
        """day[group] = dict(x = net $ per micro for the session, lo = worst MTM $ per micro). Closes the session for every account."""
        t = self.t
        for g in self.groups:
            x, lo = day[g.name]["x"], day[g.name]["lo"]
            for er in g.evals:
                if er.acct.status is None:
                    er.acct.end_day(x, lo, er.k)
            for fr in g.funded:
                if fr.acct.state == 0:
                    pays = fr.acct.end_day(x, lo, fr.k, t)
                    for pv in pays:
                        g.n_payouts += 1; g.payouts += pv; self.payouts += pv
                        if fr.n_paid_events == 0:
                            g.n_paid_accts += 1
                        fr.n_paid_events += 1
                        if fr.first_pay_t is None:
                            fr.first_pay_t = t
                        if g.first_payout_t is None:
                            g.first_payout_t = t
                        self._push_cash(t + self.payout_delay, pv)
                        self._log("payout", g.name, id=fr.id, amount=pv)
                    if fr.acct.state == 1:
                        g.died += 1
        self.net_curve.append((self.date, self.payouts - self.fees))

    # ------------------------------------------------------------------ views
    @property
    def net(self) -> float:
        return self.payouts - self.fees

    def all_funded(self) -> List[FundRec]:
        """Every funded account so far (ended and running)."""
        return list(self.retired) + [fr for g in self.groups for fr in g.funded]

    def summary(self) -> dict:
        return dict(t=self.t, date=str(self.date), cash=self.cash, fees=self.fees, payouts=self.payouts, net=self.net,
                    groups={g.name: dict(slots=g.slots, funded=len(g.funded), waiting=len(g.waiting), evals=len(g.evals), n_evals=g.n_evals, n_passed=g.n_passed, n_funded=g.n_funded, n_payouts=g.n_payouts,
                                          n_paid_accts=g.n_paid_accts, died=g.died, wasted=g.wasted, fees=g.fees, payouts=g.payouts, first_buy_t=g.first_buy_t, first_payout_t=g.first_payout_t) for g in self.groups})

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "PaperFarm":
        with open(path, "rb") as f:
            return pickle.load(f)


def run_streams(streams: Dict[str, object], i0: int, n_end: int, groups=DEFAULT_GROUPS, **kw) -> PaperFarm:
    """Batch driver (parity tests, forward tests): feed the farm session by session from stored per-micro Stream objects (real_engine.Stream) of each group."""
    farm = PaperFarm(groups, stop_new_at=n_end - 5, **kw)
    nd = {g: s.desired_micros(farm.R) for g, s in streams.items()}
    ref = next(iter(streams.values()))
    for t in range(i0, n_end):
        today = {g: dict(ok=bool(s.ok[t]), atr_usd=float(s.atr_usd[t]), nd=int(nd[g][t])) for g, s in streams.items()}
        farm.begin_session(t, ref.dates[t], today)
        farm.end_session({g: dict(x=float(s.pnl[t]), lo=float(s.low[t])) for g, s in streams.items()})
    return farm
