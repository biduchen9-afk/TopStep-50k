# Council 2, mechanics seat: what the Combine rules allow

**Position.** Nothing I tested raises EV per fee-dollar or per Combine-session reliably: R = 9,600 sits on a flat optimum (noise-area 6,400-9,600, gap 9,600). The user's idea, a lower R, raises pass probability (+5 pp noise-area, +8 pp gap-VWAP at 4,800) but lengthens Combines (1.3 to 1.7 and 1.5 to 2.4 fee cycles), so cost per pass stays near $1,000-1,300 and EV per session falls 30-40%. The ceiling for any sizing policy is only +2-3 pp per fee cycle (+5 over three) and the sleeves cannot realise it; the binding constraint is the edge, not Combine mechanics.

## Evidence
All computed by me: dev <= 2022-12-31, as-modelled costs, baseline funded cells, 854 starts every 3rd session, year-block SEs. Code: `/tmp/claude-0/-home-user-TopStep-50k/6aef8a09-9112-51df-af8b-197cbd9daa4c/scratchpad/council2/mechanics/` (`cm.py`, `exp1-6.py`, `mc.py`, `dp.py`; needs `TOPSTEP_RAW_DIR=/home/user/data_raw_live`). Parity with `run_eval_firm` checked on every start.

**1. R grid, dev** (noise / gap; EV uses the pooled funded value per pass, as raw EV swings with which years pass):

|R|2,400|4,800|9,600|19,200|
|---|---|---|---|---|
| pass % | 31/40 | 27/37 | 21/29 | 14/21 |
| median sessions | 54/91 | 19/30 | 9/15 | 6/11 |
| fee cycles | 3.6/5.6 | 1.7/2.4 | 1.3/1.5 | 1.2/1.4 |
| fees per pass $ | 2,278/2,763 | 1,260/1,270 | 1,190/1,009 | 1,666/1,297 |
| EV per session $ | 0/9 | 12/37 | 17/62 | 9/48 |

R 6,400: 24/34% pass, $1,145/$1,106 per pass; 12,800: 18/26%, $14/$61 per session. Coin flip pass 15.6, 15.7, 13.6, 10.8%, EV per try negative at every R. Bootstrap over 12 start years, best R: EV per fee-dollar 6,400 (68%) or 9,600 (29%) for noise-area, 9,600 (78%) for gap; EV per session 9,600 (57%, 62%); only EV per try, ignoring time, prefers 3,200-4,800. 2023-24 and the exact re-based cost basis give the same shape.

**2. Ceiling** (Gaussian days, Brownian-bridge lows, exact rules; dynamic programme over daily size). Driftless pass is 19-20% for any daily sd of $1.9-2.9k. Ablation at sd $2.9k: static floor, no 55% rule, no DLL 29.7%; trailing with lock 20.6%; plus DLL 19.1%; plus 55% rule 18.9% (at sd $5.6k the 55% rule costs 8.6 pp). Best size policy within 21 / 63 / 252 sessions: 20.8 / 22.4 / 22.6% driftless, 23.7 / 27.7 / 31.4% at Sharpe 0.03 per day (noise-area), 27.0 / 34.2 / 43.9% at 0.06 (gap); best fixed R: 19.0 / 20.2 / 20.2, 21.7 / 24.0 / 27.9, 24.2 / 28.9 / 41.9. The engine already matches fixed-size theory (noise 21.4% vs 21.0%, gap 28.8% vs 26.3%). The optimal table on real days lifts the coin flip +3.8 pp (2023-24 +6.4) but cuts noise-area -1.4 and gap -7 pp (3-8% timeouts): fat tails and no-signal days break it.

**3. Policies** (grid declared in `grid_declared.txt` before scoring; pass change in pp, noise / gap, SE about 1; R 9,600):
* (a) shrink once blocked, x0.5: +0.4 / +1.1; x0.25: -0.5 / 0; x0.125: -3.7 / -4.7.
* (b,c) cap 4,950 / 4,500 / 4,000 / 3,600 / 2,700: -2.1 / -3.9 / -6.7 / -9.5 / -13.8 and -2.0 / -4.1 / -6.9 / -9.4 / -15.6. At 4,950 fees fall 13-15%, EV per fee-dollar +13% / +8%, but 2023-24 pass is -5.9 / -5.3 pp.
* (d) x0.5 once the peak reaches 2,250: +4.3 / +6.4 pp (2023-24 +5.9 / +13.0) but fees +26% / +48%, EV per fee-dollar -9% / -21%, per session -24% / -34%.
* (e) reset at a 21-session boundary if balance < -$1,000 (same fee): +0.0 / +0.6 at 9,600; at 4,800 +0.7 (SE 0.2) / +2.6 (0.9), EV +$14 / +$108 per try, gap sign repeats in 2023-24. Only gap at 4,800 clears my declared bar (pass >= +2 pp, EV per fee-dollar not lower, same sign 2023-24); noise-area does not.
* 2025 noise-area: H1 6 of 42 reached target, all failed; cap 4,950 gives +1 pass, shrink x0.5 none, x0.25 +4 but -1 in H2: no rescue.

**4. Start timing** (every session): 40 shift-null tests (weekday, session of month, month, phase mod 21; two sleeves, two R, dev and 2023+): 4 below 0.05 (2 expected), none repeats; weekday spread <= 3 pp. The fee clock starts at purchase.

## Recommended next steps (by EV gain, then confidence)
1. Keep R and Combine policy. Cost 0; 85% confident the default is within noise of the best.
2. Optional operating rule, not a forward test: reset at a cycle boundary if balance < -$1,000. Gain $0-30 per try at 9,600; 60% it helps; stop if a reset costs over $199.
3. Stop Combine-mechanics research. Levers trade pass for fees; +1 pp pass is worth $23 (noise-area) to $52 (gap) per try, so the ceiling is about $50-150.

## What not to do
Lower R below 4,800; daily caps; shrinking below x0.5; early de-risking; R >= 12,800; timing rules.

## Biggest thing the chair may be wrong about
Treating pass probability as a lever. The coin flip passes 13.6-16% against 21-29% for the real sleeves, so the rules reward this edge by 8-13 pp; the 2025 pass rates (6.5-9.7% against the coin's 16.2%) are edge decay, not mechanics.

## Questions for the user
1. Is the goal EV per fee-dollar (cash-limited) or per session (slot-limited)? They differ only for lower R.
2. Does a voluntary reset cost $199 with the DLL, and count as "excessive resets"?
