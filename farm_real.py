"""Account farm on real data: up to `max_evals` concurrent Combines and `max_funded` funded accounts (Topstep: 20 / 5 per person).

Every account trades the same market history (so outcomes are correlated), differing only in start date, sizing state and,
optionally, which strategy stream it runs (accounts rotate through the streams you pass in).
Activation fee is paid when a passed Combine takes a funded slot; a passed Combine waits at most `max_wait` sessions for a slot.
New evals are only started while the expected number of funded accounts (funded + waiting + p_est x evals in flight) is below the slot count.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from contract_eval import CombineSizing
from pricing_config import EngineConfig
from real_engine import FundedPlan, RuleSet, Stream, run_combine, run_funded_detail, _lists


@dataclass(frozen=True)
class FarmSpec:
    max_evals: int = 20
    max_funded: int = 5
    max_wait: int = 21
    starts_per_day: int = 1
    p_est: float = 0.4
    # when to refill empty funded slots with new Combines: "all" = whenever the expected-funded test allows (original rule);
    # "hyst" = once full (>= max_funded funded+waiting) open nothing until `refill_dead` slots have emptied, then refill to full;
    # "buffer" = only open Combines for slots beyond the first `buffer` empty ones.
    refill: str = "all"
    refill_dead: int = 3
    buffer: int = 2
    # throttle on the farm's own recent Combine pass rate (uses only Combines already RESOLVED by day t): 0 = off.
    pause_below: float = 0.0
    th_window: int = 126          # sessions of resolved Combines to look at
    th_min: int = 4               # fewer resolved than this (after warm-up) -> probe mode
    probe_gap: int = 21           # probe mode: one Combine at a time, at least this many sessions apart


def run_farm(streams: List[Stream], sizing: CombineSizing, Rf: Optional[float], plan: FundedPlan, cfg: EngineConfig, rules: RuleSet,
             path: str, i0: int, spec: FarmSpec = FarmSpec(), n_end: Optional[int] = None, log: bool = False) -> Dict[str, float]:
    K = len(streams)
    for s in streams[1:]:
        if len(s.dates) != len(streams[0].dates) or not np.array_equal(s.dates, streams[0].dates):
            raise ValueError('all farm streams must share the same session dates (align them first)')
    n_all = min(len(s.dates) for s in streams)
    n_end = n_all if n_end is None else min(n_all, n_end)
    X = [_lists(s)[0][:n_end] for s in streams]
    L = [_lists(s)[1][:n_end] for s in streams]
    A = [s.atr_usd[:n_end].tolist() for s in streams]
    NDc = [s.desired_micros(sizing.value)[:n_end] if sizing.kind == "fixed" else [1] * n_end for s in streams]
    NDf = [s.desired_micros(Rf)[:n_end] if (Rf is not None and plan.a1 is None) else [1] * n_end for s in streams]
    dyn = sizing.value if sizing.kind == "dyn" else None
    sub, act = cfg.paths[path].monthly_cost, cfg.paths[path].activation_fee
    cyc = cfg.billing_cycle_days
    out = dict(fees=0.0, payouts=0.0, n_evals=0, n_passed=0, n_funded=0, n_wasted=0, n_payouts=0, funded_active_days=0.0,
               evals_active_days=0.0, max_funded_seen=0, max_evals_seen=0, died=0)
    evals, funded, waiting = [], [], []
    counter = 0
    events, daily = [], []
    resolved: List[tuple] = []            # (end_t, passed) of every Combine started, for the throttle
    refill_on, last_start = True, -10 ** 9
    for t in range(i0, n_end - 5):
        funded = [e for e in funded if e > t]
        keep = []
        for (end, status, si) in evals:
            if end <= t:
                if status == 1:
                    out["n_passed"] += 1
                    waiting.append((si, t))
            else:
                keep.append((end, status, si))
        evals = keep
        still = []
        for (si, since) in waiting:
            if len(funded) < spec.max_funded:
                out["fees"] += act
                out["n_funded"] += 1
                ptimes = [] if log else None
                pays, days, state, _ = run_funded_detail(X[si], L[si], NDf[si], t, rules, plan, A[si], ptimes)
                if log:
                    events.append((t, 'activation', act))
                    events.extend((pt, 'payout', pv) for pt, pv in zip(ptimes, pays))
                    events.append((t + days, 'funded_died' if state == 1 else 'funded_closed', 0.0))
                out["payouts"] += sum(pays)
                out["n_payouts"] += len(pays)
                out["died"] += int(state == 1)
                funded.append(t + days)
            elif t - since > spec.max_wait:
                out["n_wasted"] += 1
            else:
                still.append((si, since))
        waiting = still
        started = 0
        cap = spec.max_evals
        if spec.pause_below > 0:
            done = [(e, ps) for (e, ps) in resolved if e <= t]
            recent = [ps for (e, ps) in done if e > t - spec.th_window]
            warm = len(done) < spec.th_min
            if not warm and (len(recent) < spec.th_min or float(np.mean(recent)) < spec.pause_below):
                cap = 1 if t - last_start >= spec.probe_gap else 0       # probe mode
        occupied = len(funded) + len(waiting)
        if spec.refill == "hyst":
            if occupied >= spec.max_funded:
                refill_on = False
            elif occupied <= spec.max_funded - spec.refill_dead:
                refill_on = True
        gate = True if spec.refill != "buffer" else (spec.max_funded - (occupied + spec.p_est * len(evals)) > spec.buffer)
        while (len(evals) < cap and started < spec.starts_per_day and refill_on and gate
               and occupied + spec.p_est * len(evals) < spec.max_funded):
            si = counter % K
            counter += 1
            status, tau = run_combine(X[si], L[si], NDc[si], t, rules, dyn, A[si])
            out["fees"] += (1 + (max(tau, 1) - 1) // cyc) * sub
            out["n_evals"] += 1
            if log:
                for b in range(1 + (max(tau, 1) - 1) // cyc):
                    events.append((t + b * cyc, 'eval_fee', sub))
                events.append((t + tau, 'eval_pass' if status == 1 else 'eval_fail', 0.0))
            evals.append((t + tau, status, si))
            resolved.append((t + tau, float(status == 1)))
            last_start = t
            started += 1
        if log:
            daily.append((t, len(evals), len(funded)))
        out["funded_active_days"] += len(funded)
        out["evals_active_days"] += len(evals)
        out["max_funded_seen"] = max(out["max_funded_seen"], len(funded))
        out["max_evals_seen"] = max(out["max_evals_seen"], len(evals))
    days = max(n_end - 5 - i0, 1)
    out["net"] = out["payouts"] - out["fees"]
    out["avg_funded_active"] = out["funded_active_days"] / days
    out["avg_evals_active"] = out["evals_active_days"] / days
    out["sessions"] = days
    if log:
        out["events"], out["daily"] = events, daily
    return out
