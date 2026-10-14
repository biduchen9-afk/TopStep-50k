"""Independent reference implementation (equity-level state machine). Does NOT import real_engine."""
import math

LADDER = [(0, 3), (1500, 4), (2000, 5), (3000, 10), (4500, 15)]


def cap_micros(bal):
    lots = 3
    for b, l in LADDER:
        if bal >= b:
            lots = l
    return lots * 10


def ref_combine(pnl, low, nd, atr, i0, *, target=9000., mll=4500., cons=0.55, dll=None, a=None,
                maxdays=252, intraday=True, locks=True):
    """returns (status, tau)"""
    bal = 0.0            # account balance at last close
    eod_high = 0.0
    best = 0.0
    n = len(pnl)
    last = min(n, i0 + maxdays)
    for d, t in enumerate(range(i0, last), start=1):
        floor_level = min(eod_high - mll, 0.0) if locks else eod_high - mll   # equity level of MLL
        if a is None:
            k = min(nd[t], 150)
        else:
            at = atr[t]
            if at is None or math.isnan(at) or at <= 0:
                k = 1
            else:
                k = int(round(a * max(bal - floor_level, 500.0) / at))
                k = max(1, k)
                k = min(k, 150)
        day_close = pnl[t] * k
        day_low = low[t] * k
        worst_eq = bal + day_low
        dll_eq = None if dll is None else bal + dll
        booked = day_close
        if intraday:
            barrier = floor_level
            kind = 'mll'
            if dll_eq is not None and dll_eq > floor_level:
                barrier, kind = dll_eq, 'dll'
            if worst_eq <= barrier:
                if kind == 'mll':
                    return 0, d
                booked = dll
        else:
            if dll_eq is not None and worst_eq <= dll_eq:
                booked = dll
        bal += booked
        if not intraday and bal <= floor_level:
            return 0, d
        best = max(best, booked)
        eod_high = max(eod_high, bal)
        if bal >= target and best <= cons * bal:
            return 1, d
    return -1, last - i0


def ref_funded(pnl, low, nd, atr, i0, *, path='standard', t1=2000., w1=.5, r1=1., t2=None, w2=.5, r2=1.,
               max_payouts=2, a1=None, a2=None, mll=4500., dll=None, split=.9, win_day=150., pmin=125.,
               cap_std=5000., cap_con=6000., cons=.4, cdays=3, nwin=5, intraday=True):
    """returns (net payouts list, days, state, bal)  state 1 dead, 2 closed, 0 data end"""
    bal = 0.0
    eod_high = 0.0
    reset = False
    npay = 0
    wins = days_since = 0
    prof_since = best_since = 0.0
    pays = []
    n = len(pnl)
    for d, t in enumerate(range(i0, n), start=1):
        floor_level = 0.0 if reset else min(eod_high - mll, 0.0)
        if a1 is None:
            m = r1 if npay == 0 else r2
            k = max(1, int(round(nd[t] * m)))
        else:
            a = a1 if (npay == 0 or a2 is None) else a2
            at = atr[t]
            if at is None or math.isnan(at) or at <= 0:
                k = 1
            else:
                k = max(1, int(round(a * max(bal - floor_level, 250.0) / at)))
        k = min(k, cap_micros(bal))
        day_close = pnl[t] * k
        day_low = low[t] * k
        worst_eq = bal + day_low
        booked = day_close
        barrier, kind = floor_level, 'mll'
        if dll is not None and bal + dll > floor_level:
            barrier, kind = bal + dll, 'dll'
        if worst_eq <= barrier:
            if kind == 'mll':
                if intraday:
                    return pays, d, 1, floor_level
            else:
                booked = dll
        bal += booked
        if not intraday and bal <= floor_level:
            return pays, d, 1, bal
        days_since += 1
        prof_since += booked
        if booked >= win_day:
            wins += 1
        best_since = max(best_since, booked)
        if not reset:
            eod_high = max(eod_high, bal)
        if path == 'standard':
            ok = wins >= nwin
        else:
            ok = days_since >= cdays and prof_since > 0 and best_since <= cons * prof_since
        trig, frac = (t1, w1) if npay == 0 else (t2, w2)
        if ok and trig is not None and bal >= trig:
            gross = min(min(frac, .5) * bal, cap_std if path == 'standard' else cap_con)
            if gross >= pmin:
                bal -= gross
                reset = True
                wins = days_since = 0
                prof_since = best_since = 0.0
                pays.append(split * gross)
                npay += 1
                if t2 is None or npay >= max_payouts:
                    return pays, d, 2, bal
    return pays, n - i0, 0, bal
