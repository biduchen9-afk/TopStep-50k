"""Rule-parameterised evaluation and funded-account engine for different prop firms.

The Topstep specification below reproduces `real_engine.run_combine` and `real_engine.run_funded_detail` EXACTLY (checked in `test_firm_engine.py`);
other firms are expressed by changing the specification (drawdown type, trailing/static/locking, daily loss limit, consistency, minimum days, payout
eligibility, split, caps, post-payout drawdown behaviour, scaling ladder), not by changing code.

Balances are profit since the start of the stage (stage starts at 0). `lo` is the worst intraday mark-to-market of the day at ONE micro (<= 0).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from real_engine import FundedPlan, MICRO_PER_MINI, XFA_LADDER, xfa_micro_cap


@dataclass(frozen=True)
class EvalPhase:
    target: float
    dd: float
    dd_type: str = "intraday"          # 'intraday' = real-time breach incl. open P&L, 'eod' = breach tested on the end-of-day balance only
    trail: bool = True                 # False -> static floor at -dd
    lock: bool = False                 # trailing floor stops rising once it reaches `lock_level` (Topstep: starting balance)
    lock_level: float = 0.0
    dll: Optional[float] = None        # daily loss limit (negative $)
    dll_hard: bool = False             # True: touching the DLL fails the stage; False: day is liquidated at the DLL (Topstep)
    consistency: Optional[float] = None  # best day <= ratio x total profit at the moment of passing
    min_days: int = 0                  # minimum trading days (days with a non-zero result) before a pass counts
    max_days: int = 252
    max_micros: int = 150
    day_cap: Optional[float] = None    # overlay: flatten when the day's profit reaches this ($); modelled as p = min(p, cap)


@dataclass(frozen=True)
class FundedSpec:
    dd: float
    dd_type: str = "intraday"
    trail: bool = True
    lock: bool = False
    lock_level: float = 0.0
    dll: Optional[float] = None
    dll_hard: bool = False
    ladder: Tuple[Tuple[float, int], ...] = XFA_LADDER     # (minimum balance, lots); lots x 10 = micros
    split: float = 0.90
    payout_min: float = 125.0
    payout_cap: float = 5000.0
    payout_frac: float = 0.5           # max share of the balance in one request
    win_day_amt: float = 150.0         # a 'winning day' is a day with profit >= this
    win_days_req: int = 5              # winning days needed since the last payout (0 = not required)
    min_days_req: int = 0              # trading days needed since the last payout
    gap_sessions: int = 0              # minimum sessions between payouts
    consistency: Optional[float] = None  # best day <= ratio x profit since last payout (None = not required)
    after_payout: str = "zero"         # 'zero' = floor becomes 0 permanently (Topstep); 'keep' = floor unchanged; 'reset_peak' = trail restarts from the new balance
    buffer: float = 0.0                # balance that must remain after a payout
    post_floor: float = 0.0            # floor level (profit terms) after a 'zero' payout (Topstep 0; FundedNext +100)
    min_cycle_profit: float = 0.0      # profit since the last payout needed to request one (FundedNext $500)
    day_cap: Optional[float] = None    # overlay (see EvalPhase.day_cap)


@dataclass(frozen=True)
class FirmSpec:
    name: str
    eval_phases: Tuple[EvalPhase, ...]
    funded: FundedSpec
    eval_fee: float                    # price of one evaluation (per billing cycle if monthly)
    eval_monthly: bool = True          # True: billed every `cycle` sessions until the stage ends; False: one-time price
    cycle: int = 21
    activation_fee: float = 0.0
    reset_fee: float = 0.0
    max_funded: int = 5
    max_evals: int = 20
    micro_commission_rt: float = 1.22  # all-in $ per micro round turn (drives the strategy stream's costs)
    payout_delay_days: int = 3         # calendar days from request to cash (processing)
    notes: str = ""


def _floor_after(bal: float, peak: float, trail: bool, dd: float, lock: bool, lock_level: float, floor: float) -> float:
    if not trail:
        return floor
    f = peak - dd
    if lock and f > lock_level:
        f = lock_level
    return f


def run_eval_firm(phases: Tuple[EvalPhase, ...], x: List[float], lo: List[float], nd: List[int], i0: int) -> Tuple[int, int]:
    """Return (status, tau): 1 pass, 0 breach, -1 data ran out/time limit; tau = sessions used (1-based). `nd` = desired micros per session."""
    n = len(x)
    t = i0
    total_cap = sum(ph.max_days for ph in phases)
    for pi, ph in enumerate(phases):
        bal = peak = best = 0.0
        floor = -ph.dd
        days = 0
        stage_start = t
        while t < min(n, stage_start + ph.max_days):
            k = min(nd[t], ph.max_micros)
            p, w = x[t] * k, lo[t] * k
            lim = floor - bal
            if ph.dd_type == "intraday" or ph.dll is not None:
                first = lim if ph.dll is None else max(lim, ph.dll)
                if ph.dd_type == "intraday" and w <= first:
                    if ph.dll is not None and ph.dll > lim and not ph.dll_hard:
                        p = ph.dll
                    else:
                        return 0, t - i0 + 1
                elif ph.dll is not None and w <= ph.dll:
                    if ph.dll_hard:
                        return 0, t - i0 + 1
                    p = ph.dll
            if ph.day_cap is not None and p > ph.day_cap:
                p = ph.day_cap
            bal += p
            if ph.dd_type == "eod" and bal <= floor:
                return 0, t - i0 + 1
            if p != 0.0:
                days += 1
            if p > best:
                best = p
            if bal > peak:
                peak = bal
                floor = _floor_after(bal, peak, ph.trail, ph.dd, ph.lock, ph.lock_level, floor)
            t += 1
            if bal >= ph.target and days >= ph.min_days and (ph.consistency is None or best <= ph.consistency * bal):
                break
        else:
            return -1, min(n, t) - i0
        # passed this phase (loop broke)
    return 1, t - i0


def _scaled_a(scale: tuple, a1: float, bal: float, age: int, hist: List[float]) -> float:
    """Size multiple before the first payout as a function of the account's own state (balance, sessions since activation, its realised daily P&L); causal."""
    kind = scale[0]
    if kind == "ladder":                               # start at a1; use the `a` of the highest milestone the CURRENT balance has reached
        a = a1
        for m, am in scale[1]:
            if bal >= m:
                a = am
        return a
    if kind == "perf":                                 # ("perf", L, f_up, f_down): last L sessions of the account's P&L > 0 -> a1 x f_up, else a1 x f_down
        L = scale[1]
        if len(hist) < L:
            return a1
        return a1 * (scale[2] if sum(hist[-L:]) > 0 else scale[3])
    if kind == "ramp":                                 # ("ramp", d, a_start): linear from a_start to a1 over the first d sessions
        d, a0 = scale[1], scale[2]
        return a0 + (a1 - a0) * min(1.0, age / d)
    raise ValueError(kind)


def run_funded_firm(spec: FundedSpec, x: List[float], lo: List[float], nd: List[int], i0: int, plan: FundedPlan,
                    atr: Optional[List[float]] = None, times: Optional[list] = None, atr2: Optional[List[float]] = None) -> Tuple[List[float], int, int, float]:
    """Return (net payouts in order, sessions, end_state, balance): end_state 1 dead, 2 closed by plan, 0 data end."""
    bal = peak = 0.0
    floor = -spec.dd
    frozen = False                      # True after a 'zero' payout: floor fixed at 0
    stage = 1
    wins = tdays = 0
    prof = best = 0.0
    last_pay_t = -10 ** 9
    pays: List[float] = []
    hist: List[float] = []              # daily P&L of this account since activation (only used by scale = ("perf", ...))
    n = len(x)
    for t in range(i0, n):
        if plan.a1 is None:
            mult = plan.r1 if stage == 1 else plan.r2
            k = max(1, int(round(nd[t] * mult)))
        else:
            a_t = atr[t] if (stage == 1 or atr2 is None) else atr2[t]      # atr2: optional sizing unit used after the first payout (hybrid sizing)
            a = plan.a1 if (stage == 1 or plan.a2 is None) else plan.a2
            if stage == 1 and plan.scale is not None:
                a = _scaled_a(plan.scale, plan.a1, bal, t - i0, hist)
            k = 1 if not (a_t > 0) else max(1, int(round(a * max(bal - floor, 250.0) / a_t)))
        k = min(k, xfa_micro_cap(bal, spec.ladder))
        p, w = x[t] * k, lo[t] * k
        lim = floor - bal
        if spec.dd_type == "intraday" or spec.dll is not None:
            first = lim if spec.dll is None else max(lim, spec.dll)
            if spec.dd_type == "intraday" and w <= first:
                if spec.dll is not None and spec.dll > lim and not spec.dll_hard:
                    p = spec.dll
                else:
                    return pays, t - i0 + 1, 1, floor
            elif spec.dll is not None and w <= spec.dll:
                if spec.dll_hard:
                    return pays, t - i0 + 1, 1, floor
                p = spec.dll
        if spec.day_cap is not None and p > spec.day_cap:
            p = spec.day_cap
        bal += p
        hist.append(p)
        if spec.dd_type == "eod" and bal <= floor:
            return pays, t - i0 + 1, 1, bal
        if p != 0.0:
            tdays += 1
        prof += p
        if p >= spec.win_day_amt:
            wins += 1
        if p > best:
            best = p
        if not frozen and bal > peak:
            peak = bal
            floor = _floor_after(bal, peak, spec.trail, spec.dd, spec.lock, spec.lock_level, floor)
        eligible = (spec.win_days_req == 0 or wins >= spec.win_days_req) and tdays >= spec.min_days_req and (t - last_pay_t) >= spec.gap_sessions
        if spec.consistency is not None:
            eligible = eligible and prof > 0 and best <= spec.consistency * prof
        if spec.min_cycle_profit > 0:
            eligible = eligible and prof >= spec.min_cycle_profit
        trig, frac = (plan.t1, plan.w1) if stage == 1 else (plan.t2, plan.w2)
        if eligible and bal >= trig:
            gross = min(min(frac, spec.payout_frac) * bal, spec.payout_cap, bal - spec.buffer)
            if gross >= spec.payout_min:
                bal -= gross
                if spec.after_payout == "zero":
                    floor, frozen = spec.post_floor, True
                elif spec.after_payout == "reset_peak":
                    peak = bal
                    floor = bal - spec.dd
                wins = tdays = 0
                prof = best = 0.0
                last_pay_t = t
                pays.append(spec.split * gross)
                if times is not None:
                    times.append(t)
                stage = 2
                if plan.t2 is None or len(pays) >= plan.max_payouts:
                    return pays, t - i0 + 1, 2, bal
    return pays, n - i0, 0, bal


def topstep_150k(path: str = "no_activation_dll", payout_path: str = "standard") -> FirmSpec:
    """Topstep 150K as modelled so far (config.yaml). `path` picks the fee path."""
    from pricing_config import load_config
    from real_engine import rules_from_config
    cfg = load_config(); r = rules_from_config(cfg, path); p = cfg.paths[path]
    ph = EvalPhase(target=r.profit_target, dd=r.mll, dd_type="intraday" if r.intraday_mll else "eod", trail=True, lock=r.combine_mll_locks, lock_level=0.0,
                   dll=r.dll, dll_hard=False, consistency=r.consistency_ratio, min_days=0, max_days=r.max_combine_days, max_micros=r.combine_micros)
    fd = FundedSpec(dd=r.mll, dd_type="intraday" if r.intraday_mll else "eod", trail=True, lock=True, lock_level=0.0, dll=r.dll, dll_hard=False, ladder=r.ladder,
                    split=r.split, payout_min=r.payout_min, payout_cap=r.cap_standard if payout_path == "standard" else r.cap_consistency, payout_frac=0.5,
                    win_day_amt=r.win_day, win_days_req=r.standard_win_days if payout_path == "standard" else 0,
                    min_days_req=0 if payout_path == "standard" else r.consistency_days, consistency=None if payout_path == "standard" else r.consistency_xfa, after_payout="zero")
    return FirmSpec("Topstep 150K", (ph,), fd, eval_fee=p.monthly_cost, eval_monthly=True, cycle=cfg.billing_cycle_days, activation_fee=p.activation_fee, reset_fee=p.reset_fee,
                    max_funded=5, max_evals=20, micro_commission_rt=cfg.friction.micro_round_turn_commission, payout_delay_days=3)
