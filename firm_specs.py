"""Rule specifications for the prop firms, as found by web search on 2026-10-14 (WebFetch to the firms' own pages was blocked, so almost everything comes from
search snippets of official pages and third-party reviews: treat every number as UNVERIFIED until checked on the firm's own site / support).

Each entry has `conf` (what is solid vs not) and `flags` (rules that the simulation cannot enforce, or that could make the plan impossible).
Assumptions made where a number was not found are listed in `assumed`. All firms use the SAME strategy stream (Topstep cost model: $1.22 per micro round
turn + 1 tick), the Topstep session calendar, 3 calendar days payout delay. Intraday-trailing drawdowns are trailed on end-of-day highs (we do not store the
intraday high), which is slightly lenient.
"""
from __future__ import annotations

from typing import Dict

from firm_engine import EvalPhase, FirmSpec, FundedSpec, topstep_150k

L15 = ((0.0, 15),)


def specs() -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    ts = topstep_150k("no_activation_dll")
    out["Topstep 150K"] = dict(spec=ts, t1_grid=(3000., 4000.), max_payouts=99, conf="best known (help-centre pages via search, Oct 2026)",
                               flags=["personal device only, no VPS (third-party summary of API terms)", "native trade copier exists for Combine/XFA"], assumed=[])
    out["FTMO Growth 150K"] = dict(
        spec=FirmSpec("FTMO Growth 150K", (EvalPhase(9000., 5000., "eod", True, True, 0.0, None, False, 0.40, 0, 252, 150),),
                      FundedSpec(dd=5000., dd_type="eod", lock=True, lock_level=0.0, dll=-3000., dll_hard=False, ladder=((0., 6), (1500., 8), (3000., 10), (4500., 12), (6000., 15)),
                                 split=0.90, payout_min=100., payout_cap=4000., payout_frac=0.5, win_day_amt=200., win_days_req=4, after_payout="zero"),
                      eval_fee=229., eval_monthly=True, max_funded=3, max_evals=10),
        t1_grid=(1500., 3000., 4500.), max_payouts=99,
        conf="Beta product (launched Aug-Sep 2026); drawdown/target/consistency LIKELY-CONFIRMED, price LIKELY, qualifying-day size and scaling ladder NOT FOUND",
        flags=["FTMO Futures is Beta; terms may change", "max 3 sim-funded accounts at once", "automation/EA policy conflicting", "Cambodia eligibility not found"],
        assumed=["qualifying day = profit >= $200", "contract ladder 6->15 minis in 5 steps", "no eval contract cap below 15 minis", "payout min $100", "drawdown after payout floored at 0"])
    out["FTMO Pro 150K"] = dict(
        spec=FirmSpec("FTMO Pro 150K", (EvalPhase(9000., 6000., "eod", True, True, 0.0, -2000., True, 0.50, 0, 252, 150),),
                      FundedSpec(dd=6000., dd_type="eod", lock=True, lock_level=0.0, dll=-2000., dll_hard=True, ladder=((0., 6), (1500., 8), (3000., 10), (4500., 12), (6000., 15)),
                                 split=0.90, payout_min=100., payout_cap=8000., payout_frac=1.0, win_day_amt=300., win_days_req=5, after_payout="zero"),
                      eval_fee=259., eval_monthly=True, max_funded=3, max_evals=10),
        t1_grid=(2000., 4000., 8000.), max_payouts=99,
        conf="price conflicting ($259/$269), hard DLL $2,000 and $6,000 drawdown LIKELY, payout details LIKELY",
        flags=["hard $2,000 daily loss limit applies in funded stage too", "max 3 sim-funded accounts", "Beta"],
        assumed=["qualifying day = profit >= $300", "ladder as Growth"])
    out["FundedNext Flex 150K"] = dict(
        spec=FirmSpec("FundedNext Flex 150K", (EvalPhase(8000., 4000., "eod", True, True, 100.0, None, False, 0.40, 0, 252, 80),),
                      FundedSpec(dd=4000., dd_type="eod", lock=True, lock_level=100.0, dll=None, ladder=((0., 8),), split=0.80, payout_min=250., payout_cap=4000., payout_frac=0.5,
                                 win_day_amt=250., win_days_req=5, after_payout="zero", post_floor=100.0, min_cycle_profit=500.),
                      eval_fee=250., eval_monthly=False, max_funded=5, max_evals=5),
        t1_grid=(1500., 3000., 4500.), max_payouts=5,
        conf="eval/funded rules CONFIRMED by help-centre snippets; current price and reset fee NOT FOUND; split 80% base (95% per some sources)",
        flags=["official help article bans copying between FundedNext accounts, even your own", "micro-scalping rule: trades held < 10 seconds", "payouts to Cambodia: bank transfer not available (USDT/USDC/Rise)",
               "account concludes after the 5th payout", "closes after 30 days inactivity"],
        assumed=["eval price $250 one-time (launch listing; unknown whether monthly)", "split 80%", "funded contract cap 8 minis"])
    out["Alpha Advanced 150K"] = dict(
        spec=FirmSpec("Alpha Advanced 150K", (EvalPhase(12000., 5250., "eod", True, True, 0.0, None, False, 0.40, 3, 252, 150),),
                      FundedSpec(dd=5250., dd_type="eod", lock=True, lock_level=0.0, dll=None, ladder=L15, split=0.90, payout_min=1000., payout_cap=15000., payout_frac=0.5,
                                 win_day_amt=200., win_days_req=5, gap_sessions=5, after_payout="keep"),
                      eval_fee=489., eval_monthly=True, max_funded=3, max_evals=10),
        t1_grid=(2500., 4000., 6000.), max_payouts=99,
        conf="price/target/drawdown LIKELY; payout rules CONFIRMED (help centre)",
        flags=["bots/EAs/AI BANNED on all plans (semi-automated allowed if you click each trade)", "counterparty risk: NinjaTrader/Tradovate exit Jul 2026, Premium accounts closed, payouts in batches",
               "official page: 3 qualified accounts (third parties say 5)", "hedging banned"],
        assumed=["contract cap 15 minis", "no daily loss limit"])
    out["Alpha Standard 150K"] = dict(
        spec=FirmSpec("Alpha Standard 150K", (EvalPhase(9000., 4500., "eod", True, True, 0.0, None, False, 0.50, 3, 252, 150),),
                      FundedSpec(dd=4500., dd_type="eod", lock=True, lock_level=0.0, dll=-3000., dll_hard=False, ladder=L15, split=0.90, payout_min=250., payout_cap=5000., payout_frac=0.5,
                                 win_day_amt=200., win_days_req=5, gap_sessions=5, consistency=0.40, after_payout="keep"),
                      eval_fee=349., eval_monthly=True, max_funded=3, max_evals=10),
        t1_grid=(1500., 3000., 4500.), max_payouts=99,
        conf="UNCERTAIN whether currently sold; quoted by one comparator citing official pages",
        flags=["bots/EAs/AI banned", "counterparty risk (see Advanced)"],
        assumed=["EOD trailing", "min payout $250", "winning day $200 x 5", "min 3 days"])
    out["Take Profit Trader PRO 150K"] = dict(
        spec=FirmSpec("Take Profit Trader PRO 150K", (EvalPhase(9000., 4500., "eod", True, True, 0.0, None, False, 0.50, 3, 252, 150),),
                      FundedSpec(dd=4500., dd_type="intraday", lock=True, lock_level=0.0, dll=None, ladder=L15, split=0.80, payout_min=250., payout_cap=1e9, payout_frac=1.0,
                                 win_day_amt=0., win_days_req=0, after_payout="keep", buffer=4500.),
                      eval_fee=360., eval_monthly=True, activation_fee=130., max_funded=5, max_evals=10),
        t1_grid=(4750., 5500., 7000.), max_payouts=99,
        conf="all figures from third-party review sites; official help centre never opened",
        flags=["copy trading banned on PRO", "bots/automation banned or restricted (uncertain)", "must trade at least once per calendar week (uncertain)", "Sustainable Trading Policy lets the firm restrict aggressive traders",
               "payout of <= $250 carries a $50 fee (ignored)"],
        assumed=["list price $360/mo (promo codes of ~40% are common: not assumed)", "intraday trailing approximated on EOD highs", "buffer $4,500 must remain after every payout"])
    return out
