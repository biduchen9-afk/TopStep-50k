"""Multi-firm account farm with a self-funding rule.

Cash discipline ("out of pocket only for the first Topstep account"):
  * Until the FIRST payout (from any firm) is received, only the lead firm (first in the list) may buy evaluations, at most `k0` at a time; its fees are
    paid out of pocket (counted in `oop_total`).
  * After the first payout nothing more is injected: any purchase (any firm) needs  cash >= price + reserve,  with
    cash = payouts received (after the firm's payout delay) - every fee paid so far (+ the out-of-pocket money injected before the first payout), and
    reserve = `reserve_cycles` billing cycles of the monthly fees of all open evaluations including the new one.
  * A non-lead firm may buy only after the previous firm in the list has received a payout (`sequential_unlock`).
Every account of a firm runs one of that firm's streams (round-robin); all firms share the same session calendar (same dates).
Evaluation outcomes are simulated at purchase (standard); decisions only use information available at that day (cash received so far, accounts open).
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from firm_engine import FirmSpec, run_eval_firm, run_funded_firm
from real_engine import FundedPlan, Stream


@dataclass(frozen=True)
class FirmPlan:
    spec: FirmSpec
    streams: Sequence[Stream]
    R: float                        # evaluation risk unit ($ per ATR unit)
    plan: FundedPlan                # funded sizing / payout triggers
    p_est: float = 1.0
    refill: str = "hyst"            # 'all' | 'hyst'
    refill_dead: int = 3
    max_wait: int = 21
    starts_per_day: int = 1
    funded_atr: Optional[Sequence[np.ndarray]] = None   # per stream: replaces atr_usd in the FUNDED sizing rule only (Combine sizing keeps the stream's own ATR), e.g. Kelly-equivalent unit
    firm: Optional[str] = None      # several entries (account groups) can belong to one firm; None = the entry is its own firm


@dataclass(frozen=True)
class Policy:
    k0: int = 1                     # lead-firm evaluations open at once before the first payout
    reserve_cycles: float = 3.0     # None -> no cash gating at all (unconstrained, used to test against run_farm)
    sequential_unlock: bool = True
    global_one_eval: bool = False   # at most ONE open evaluation across ALL firms/lists; the next purchase goes to the list with the most empty funded slots (ties: list order)
    group_aware: bool = False       # firm-level gating (unlock, out-of-pocket phase, maintenance slot) for entries that share a firm; candidates are filtered BEFORE the single global queue picks one
    stagger_gap: int = 0            # minimum sessions between two evaluation purchases of the same firm (decorrelates attempts that share one market path)
    unlock_cash: float = 0.0        # a non-lead firm may buy only when cash >= unlock_cash + price + reserve ('only once our payouts are enough')
    single_pipeline: bool = True   # before the first payout the lead firm runs ONE account at a time (Combine -> pass -> funded), as in 'out of pocket only for the first account'
    maintenance: bool = True        # lead firm keeps `k0` evaluation(s) running whenever cash >= price (no reserve); the reserve only gates EXPANSION
    payout_delay_sessions: int = 2


def _sessions(days: int) -> int:
    return max(1, int(round(days * 252 / 365)))


def run_multifirm(firms: List[FirmPlan], pol: Policy, i0: int, n_end: Optional[int] = None, log: bool = False) -> Dict:
    d0 = firms[0].streams[0].dates
    for f in firms:
        for s in f.streams:
            if len(s.dates) != len(d0) or not np.array_equal(s.dates, d0):
                raise ValueError("all streams must share the same session dates")
    n_all = len(d0)
    n_end = n_all if n_end is None else min(n_all, n_end)
    F = []
    for fp in firms:
        X = [s.pnl[:n_end].tolist() for s in fp.streams]
        L = [s.low[:n_end].tolist() for s in fp.streams]
        A = [(fp.funded_atr[k] if fp.funded_atr is not None else s.atr_usd)[:n_end].tolist() for k, s in enumerate(fp.streams)]
        ND = [s.desired_micros(fp.R)[:n_end] for s in fp.streams]
        F.append(dict(fp=fp, X=X, L=L, A=A, ND=ND, evals=[], funded=[], waiting=[], counter=0, refill_on=True, n_paid_events=0, first_payout_t=None,
                      fees=0.0, payouts=0.0, n_paid_accts=0, n_evals=0, n_passed=0, n_funded=0, n_payouts=0, died=0, wasted=0, first_buy_t=None, first_pass_t=None, first_pass_start=None))
    cash_events: List[tuple] = []          # (t, amount) heap: payouts arriving and scheduled fee debits (negative)
    cash = 0.0
    oop = 0.0
    first_payout_received = False
    min_cash = 0.0
    shortfall_days = 0
    ruined_t = None
    events: List[tuple] = []
    daily = []
    gated = pol.reserve_cycles is not None
    full_fees = 0.0
    for t in range(i0, n_end - 5):
        while cash_events and cash_events[0][0] <= t:
            _, amt = heapq.heappop(cash_events)
            cash += amt
            if amt > 0:
                first_payout_received = True
        min_cash = min(min_cash, cash)
        if gated and first_payout_received and ruined_t is None:
            lead_idx = [j for j, g in enumerate(F) if (g["fp"].firm or g["fp"].spec.name) == (F[0]["fp"].firm or F[0]["fp"].spec.name)] if pol.group_aware else [0]
            if not any(F[j]['evals'] or F[j]['funded'] or F[j]['waiting'] for j in lead_idx) and cash < F[0]['fp'].spec.eval_fee:
                ruined_t = t
        if cash < 0:
            shortfall_days += 1
        for fi, f in enumerate(F):
            fp, spec = f["fp"], f["fp"].spec
            f["funded"] = [e for e in f["funded"] if e > t]
            keep = []
            for (end, status, si, t_start) in f["evals"]:
                if end <= t:
                    if status == 1:
                        f["n_passed"] += 1
                        f["waiting"].append((si, t, t_start))
                        if f["first_pass_t"] is None:
                            f["first_pass_t"], f["first_pass_start"] = t, t_start
                else:
                    keep.append((end, status, si, t_start))
            f["evals"] = keep
            still = []
            for (si, since, t_start) in f["waiting"]:
                if len(f["funded"]) < spec.max_funded:
                    fee = spec.activation_fee
                    f["fees"] += fee; full_fees += fee; cash -= fee
                    f["n_funded"] += 1
                    tm: list = []
                    pays, days, state, _ = run_funded_firm(spec.funded, f["X"][si], f["L"][si], f["ND"][si], t, fp.plan, f["A"][si], tm)
                    for pt, pv in zip(tm, pays):
                        heapq.heappush(cash_events, (pt + pol.payout_delay_sessions, pv))
                        if f["first_payout_t"] is None:
                            f["first_payout_t"] = pt
                    if log:
                        events.append((t, spec.name, "activation", fee))
                        events.extend((pt, spec.name, "payout", pv) for pt, pv in zip(tm, pays))
                        events.append((t + days, spec.name, "funded_died" if state == 1 else "funded_closed", 0.0))
                    f["payouts"] += sum(pays); f["n_payouts"] += len(pays); f["n_paid_accts"] += int(len(pays) > 0); f["died"] += int(state == 1)
                    f["funded"].append(t + days)
                elif t - since > fp.max_wait:
                    f["wasted"] += 1
                else:
                    still.append((si, since, t_start))
            f["waiting"] = still
        # ---- purchases
        pick = None
        if pol.global_one_eval:
            if any(g["evals"] for g in F):
                pick = -1
            else:
                vac = [(g["fp"].spec.max_funded - (len(g["funded"]) + len(g["waiting"])), -gi) for gi, g in enumerate(F)]
                best = max(vac)
                pick = -best[1] if best[0] > 0 else -1
        burn = sum(len(g["evals"]) * g["fp"].spec.eval_fee for g in F if g["fp"].spec.eval_monthly)
        if pol.group_aware:
            firm_of = [g["fp"].firm or g["fp"].spec.name for g in F]
            firm_order = list(dict.fromkeys(firm_of)); lead = firm_of[0]
            entries_of = {fm: [j for j, x in enumerate(firm_of) if x == fm] for fm in firm_order}
            cand = []
            any_open = any(g["evals"] for g in F)
            for fi, f in enumerate(F):
                fp, spec = f["fp"], f["fp"].spec; fm = firm_of[fi]; is_lead = fm == lead
                if len(f["evals"]) >= spec.max_evals or (pol.global_one_eval and any_open):
                    continue
                if not is_lead:
                    if not first_payout_received:
                        continue
                    prev = firm_order[firm_order.index(fm) - 1]
                    if pol.sequential_unlock and not any(F[j]["first_payout_t"] is not None for j in entries_of[prev]):
                        continue
                if is_lead and gated and not first_payout_received and pol.single_pipeline and any(F[j]["evals"] or F[j]["funded"] or F[j]["waiting"] for j in entries_of[lead]):
                    continue
                occupied = len(f["funded"]) + len(f["waiting"])
                vacancy = spec.max_funded - occupied
                if vacancy <= 0 or occupied + fp.p_est * len(f["evals"]) >= spec.max_funded:
                    continue
                price = spec.eval_fee
                if gated:
                    reserve = pol.reserve_cycles * (burn + (price if spec.eval_monthly else 0.0))
                    lead_open = sum(len(F[j]["evals"]) for j in entries_of[lead])
                    if is_lead and not first_payout_received:
                        pass
                    elif pol.maintenance and is_lead and lead_open < pol.k0 and cash >= price:
                        pass
                    elif cash < price + reserve + (0.0 if is_lead else pol.unlock_cash):
                        continue
                cand.append((vacancy, -fi))
            picks = [-max(cand)[1]] if (cand and pol.global_one_eval) else [-c[1] for c in cand]
            for fi in picks:
                f = F[fi]; fp, spec = f["fp"], f["fp"].spec; price = spec.eval_fee
                if gated and firm_of[fi] == lead and not first_payout_received and cash < price:
                    oop += price - cash; cash = price
                si = f["counter"] % len(fp.streams); f["counter"] += 1
                status, tau = run_eval_firm(spec.eval_phases, f["X"][si], f["L"][si], f["ND"][si], t)
                bills = (1 + (max(tau, 1) - 1) // spec.cycle) if spec.eval_monthly else 1
                f["fees"] += bills * price; full_fees += bills * price; cash -= price
                for b in range(1, bills):
                    heapq.heappush(cash_events, (t + b * spec.cycle, -price))
                f["n_evals"] += 1
                if f["first_buy_t"] is None:
                    f["first_buy_t"] = t
                f["last_buy_t"] = t
                if log:
                    events.append((t, spec.name, "eval_fee", price))
                    events.extend((t + b * spec.cycle, spec.name, "eval_fee", price) for b in range(1, bills))
                    events.append((t + tau, spec.name, "eval_pass" if status == 1 else "eval_fail", 0.0))
                f["evals"].append((t + tau, status, si, t))
        for fi, f in (enumerate(F) if not pol.group_aware else ()):
            fp, spec = f["fp"], f["fp"].spec
            if pick is not None and fi != pick:
                continue
            if gated:
                if fi > 0:
                    if not first_payout_received:
                        continue
                    if pol.sequential_unlock and F[fi - 1]["first_payout_t"] is None:
                        continue
                cap = spec.max_evals
                if fi == 0 and not first_payout_received:
                    cap = min(cap, pol.k0)
            else:
                cap = spec.max_evals
            occupied = len(f["funded"]) + len(f["waiting"])
            if gated and fi == 0 and not first_payout_received and pol.single_pipeline and (f["evals"] or f["funded"] or f["waiting"]):
                continue
            if fp.refill == "hyst":
                if occupied >= spec.max_funded:
                    f["refill_on"] = False
                elif occupied <= spec.max_funded - fp.refill_dead:
                    f["refill_on"] = True
            started = 0
            if pol.stagger_gap and f.get("last_buy_t") is not None and t - f["last_buy_t"] < pol.stagger_gap:
                continue
            while (len(f["evals"]) < cap and started < fp.starts_per_day and f["refill_on"]
                   and occupied + fp.p_est * len(f["evals"]) < spec.max_funded):
                si = f["counter"] % len(fp.streams)
                price = spec.eval_fee
                if gated:
                    new_burn = burn + (price if spec.eval_monthly else 0.0)
                    reserve = pol.reserve_cycles * new_burn
                    if fi == 0 and not first_payout_received:
                        if cash < price:                          # out of pocket: top up exactly what is missing
                            oop += price - cash
                            cash = price
                    elif pol.maintenance and fi == 0 and len(f["evals"]) < pol.k0 and cash >= price:
                        pass                                         # maintenance slot: self-funded, reserve not required
                    elif cash < price + reserve + (pol.unlock_cash if fi > 0 else 0.0):
                        break
                f["counter"] += 1
                status, tau = run_eval_firm(spec.eval_phases, f["X"][si], f["L"][si], f["ND"][si], t)
                bills = (1 + (max(tau, 1) - 1) // spec.cycle) if spec.eval_monthly else 1
                f["fees"] += bills * price; full_fees += bills * price
                cash -= price
                for b in range(1, bills):
                    heapq.heappush(cash_events, (t + b * spec.cycle, -price))
                if spec.eval_monthly:
                    burn += price
                f["n_evals"] += 1
                if f["first_buy_t"] is None:
                    f["first_buy_t"] = t
                f["last_buy_t"] = t
                if log:
                    events.append((t, spec.name, "eval_fee", price))
                    events.extend((t + b * spec.cycle, spec.name, "eval_fee", price) for b in range(1, bills))
                    events.append((t + tau, spec.name, "eval_pass" if status == 1 else "eval_fail", 0.0))
                f["evals"].append((t + tau, status, si, t))
                started += 1
        if log:
            daily.append((t, cash, tuple((len(g["evals"]), len(g["funded"])) for g in F)))
    out = dict(ruined_t=ruined_t, oop_total=oop, min_cash=min_cash, shortfall_days=shortfall_days, sessions=max(n_end - 5 - i0, 1), end_cash=cash,
               fees=sum(f["fees"] for f in F), payouts=sum(f["payouts"] for f in F))
    out["net"] = out["payouts"] - out["fees"]
    out["firms"] = {f["fp"].spec.name: dict(fees=f["fees"], payouts=f["payouts"], net=f["payouts"] - f["fees"], n_evals=f["n_evals"], n_passed=f["n_passed"],
                                            n_funded=f["n_funded"], n_payouts=f["n_payouts"], n_paid_accts=f["n_paid_accts"], died=f["died"], wasted=f["wasted"], first_buy_t=f["first_buy_t"],
                                            first_pass_t=f["first_pass_t"], first_pass_start=f["first_pass_start"], first_payout_t=f["first_payout_t"]) for f in F}
    if log:
        out["events"], out["daily"] = events, daily
    return out
