# Council 2, farm-and-capital architect

**Position.** Keep buying Combines in the paper farm (free; the only new information); do not run a real farm. It needs pass rate x payout per funded account >= about $260 per Combine; since December 2024 it paid about $19 per Combine against $279 of fees. A seed-capped (<= $2,000) first-account pilot is a gated decision, not advised now: about 15-20% chance of ending ahead, typical result -$2k. Pause rules limit damage, create no profit; a money cap protects better.

**Evidence.** [C] computed by me (repaired data, 3+2 baseline, rules v2); scripts in `/tmp/claude-0/-home-user-TopStep-50k/6aef8a09-9112-51df-af8b-197cbd9daa4c/scratchpad/council2/farm/` (`gfarm.py` = my gated `run_multifirm`, identical on 247 farms). [R] read.

1. **Break-even** [C]: p* = fees per Combine / payout per funded account (Combine every 3 sessions).

| Starts | pass (farm mix) | fees | payout per funded | p* |
|---|---|---|---|---|
| dev 2011-22, modelled | 22.8% | $262 | $3.1k | 8.4% |
| dev, re-based exact | 26.9% | $261 | $4.5k | 5.8% |
| 2023-24 | 29.8% | $241 | $3.6k | 6.7% |
| since 2024-12 | 7.1% | $279 | $0.26k (about 2 independent funded accounts) | none; 8.9% at dev payout |

Dev 2011-16 p* is 27.5%, 2017-22 5.8%. Coin flip: pass 13.6 / 15.3 / 16.2%, own p* 19.9 / 11.8 / 13.0% (dev / 2023-24 / since 2024-12; EV per Combine +$21, SD $107 over 40 seeds). It sits at break-even; "beat 14-16%" says nothing about profit. Since 2024-12 (<= 2 passes in about 23 Combines): likelihood 0.03 at p=27%, 0.31 at 15%, 0.79 at 7%.

2. **Pause rules** [C], 247 dev farms; coin = 988 zero-edge farms; seen = 250-session farms from 2024-12..07 (16) and 2023-01..2024-06 (39), descriptive.

| Rule | fees | mean | p25 | worst | d mean modelled / exact | coin worst | seen bad / good |
|---|---|---|---|---|---|---|---|
| none | 13.7k | 12.7k | -4.7k | -11.6k | | -27.2k | -4.4 / +7.6k |
| 6 failures in a row, pause 126 sessions | 8.6k | 11.7k | -5.8k | -8.0k | -1.0 / -13.8k | -11.3k | -3.1 / +7.3k |
| 8 in a row, pause 252 | 8.9k | 12.2k | -5.0k | -8.2k | -0.6 / -7.0k | -11.5k | -3.3 / +7.1k |
| 8 in a row, resume on 2 of 8 shadow passes | 11.0k | 9.1k | -7.7k | -10.9k | -3.7 / -2.7k | -19.1k | -3.3 / +7.6k |
| calendar, <= 3 per quarter | 8.3k | 7.3k | -3.9k | -8.8k | -5.4 / -20.9k | -8.4k | -2.8 / +4.9k |
| calendar, <= 2 per quarter | 5.8k | 4.4k | -3.2k | -7.0k | -8.3 / -26.4k | -6.0k | -1.8 / +3.3k |

No rule of 13 raised the mean beyond noise (best +$0.2k, almost never firing). The two timer rules cut fees ~35% and the worst farm ~30% but cost $0.6-1.0k (modelled) or $7-14k (exact); most pauses are false alarms (62% of dev farms hold 8+ failed Combines in a row). At ~$8.7k fees they cost less than the calendar cap ($5.4k / $20.9k), keep the 2023-24 profit and save alike in the bad window; a plain cap beats the shadow rule; with no regime (coin) it holds the tail better. No robust winner.

3. **Money at risk**, seed $2,000, one pipeline [C]: P(first payout within the seed) 61% (dev modelled), 81% (exact), 38% (coin flip), 87% (starts 2023-01..2024-06), 1 of 17 (starts 2024-12..07); median net -1.0k, +9.4k, -2.0k, +10.3k, -2.0k. A first payout is weak evidence.

4. **Conduct** [C]: 18 Combines a year (up to 6 in a month), 77% fail, 9.4 funded deaths per farm; sessions with >= 2 MLL hits 3.0 per farm (94% of farms), >= 2 funded deaths together 1.2 (77%). Noise and gap hold opposite positions in 1 of 4,123 sessions [R].

**Decision gates (define, do not act)**
- G0, clock: forward test VALID (>= 98% complete sessions); today invalid. Risk $0.
- G1, one real Combine becomes eligible: >= 36 time-disjoint paper Combines (2.2 years), >= 9 passes, paper payout per funded account >= $2k, no PR18 STOP, holds at $250 overshoot. False go 8% (coin), 0.1% (p=7%); power 67% (p=27%). Retire at <= 4 passes (90% if p=7%, 36% if 15%, 2% if 27%); 5-8 keep going. Risk: seed $2,000; a wrong go costs about $2k, a wrong stop forgoes +$9k to +$21k.
- G2, beyond one account: >= 3 payouts from >= 2 accounts, measured overshoot < 3 ticks per contract.

**Recommended next steps**
1. Paper farm, one Combine at a time, G1 registered now. Cost: always-on machine. Go within 2.5 years: about 20% (prior 25%). Stop: retire rule.
2. Four written Topstep questions (purchase cadence, RTP triggers, one-Account wording, call-up). Free.
3. Any real attempt: money cap first. Conduct mitigations: streak pause (8 in a row, 126 sessions; -$3.2k modelled, -$2.5k exact); cadence <= 1 Combine per 21 sessions (2-hit days 3.0 to 1.1; -$5.0k / -$21k); opposite-position veto; signals from Topstep's own feed (external feeds listed prohibited); payouts capped at 5 per account for call-up (exact mean $30.7k) [R]. Stacking and "excessive purchases" have no stated threshold; RTP (second-hand) can force Consistency: exact mean $38.8k to $6.5k [R].

**What not to do.** Run a continuous real farm; gate on first payout or farm net (coin farm: mean $10k, median $2.3k, per-seed means -$1k to +$25k); read pass >15% as profit; add concurrency.

**Biggest thing the chair may be wrong about.** The coin-flip benchmark is not the decision metric: break-even swings with funded productivity, and paper cannot measure fills or Topstep's response. A $2k-capped pilot has positive mean EV above P(good) of 8-18%; the no-pilot rule costs that information.

**Questions for the user.** What seed loss is acceptable? Is waiting 2+ years for G1 acceptable?
