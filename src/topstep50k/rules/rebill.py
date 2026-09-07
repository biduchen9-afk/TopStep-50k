"""Combine subscription lifecycle: rebills and Reset Credits.

This is account-management bookkeeping, separate from the trading-rule
engine in `topstep.py` -- it tracks the monthly subscription cadence and
the Reset Credit bank a trader accumulates, so a live-automation script
can answer "do I have a free reset available right now" or "when is my
next payment due" without guessing.

Two distinct actions, often confused:
  * `rebill()`  -- paying to keep the Combine subscription active. Adds
    ONE Reset Credit to the bank. Does NOT touch account progress.
  * `redeem_reset_credit()` -- spending a credit to wipe the Combine
    back to starting conditions (a deliberate trader choice, e.g. after
    an MLL breach, to retry without buying a new account). ALSO pushes
    the next rebill date out 30 days, as a side benefit.

Simply continuing to pay (rebilling) without ever redeeming a credit
preserves 100% of whatever profit/progress the account currently has --
there is no special "keep some fixed profit amount" mechanic tied to a
dollar threshold; that was a misreading of how rebilling works (see the
2026-08-26 chat discussion this module was built from).

Sourced from Topstep's help-center "What is a Reset?" article via
third-party 2026 aggregators (help.topstep.com is blocked from this
sandbox's network policy; not independently re-verified against the
primary source -- see docs/rules_sources.md). Confirm before relying on
this for a real account:
  * Credit expiry: credits issued 2025-12-11 or later expire 1 year
    after issue; credits issued before that date do not expire.
  * Credits are tied to one specific account size + type; cannot be
    combined, split, or transferred between accounts.
  * Limit of 2 redemptions per account per day.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

REBILL_INTERVAL_DAYS = 30
CREDIT_EXPIRY_POLICY_START = date(2025, 12, 11)
CREDIT_EXPIRY_DAYS = 365
MAX_REDEMPTIONS_PER_DAY = 2


@dataclass
class ResetCredit:
    issued_on: date
    expires_on: date | None  # None = does not expire (issued before the policy start date)
    used: bool = False
    used_on: date | None = None

    def is_available(self, as_of: date) -> bool:
        if self.used:
            return False
        return not (self.expires_on is not None and as_of >= self.expires_on)


@dataclass
class RebillLifecycle:
    """Tracks one Combine subscription's rebill cadence and Reset Credit
    bank. Construct with the account's current rebill date; call
    `rebill()` each time a payment is made and `redeem_reset_credit()`
    each time the trader chooses to use a credit.
    """

    account_size: str   # e.g. "50K"
    account_type: str = "combine"
    rebill_date: date = field(default_factory=date.today)
    credits: list[ResetCredit] = field(default_factory=list)
    redemptions_today: dict[date, int] = field(default_factory=dict)

    def rebill(self, on_date: date) -> ResetCredit:
        """Pay for another cycle. Adds one Reset Credit; advances the
        rebill date by REBILL_INTERVAL_DAYS. Returns the new credit."""
        expires = (
            on_date + timedelta(days=CREDIT_EXPIRY_DAYS)
            if on_date >= CREDIT_EXPIRY_POLICY_START
            else None
        )
        credit = ResetCredit(issued_on=on_date, expires_on=expires)
        self.credits.append(credit)
        self.rebill_date = on_date + timedelta(days=REBILL_INTERVAL_DAYS)
        return credit

    def available_credits(self, as_of: date) -> list[ResetCredit]:
        return [c for c in self.credits if c.is_available(as_of)]

    def n_available_credits(self, as_of: date) -> int:
        return len(self.available_credits(as_of))

    def redeem_reset_credit(self, on_date: date) -> ResetCredit:
        """Spend the credit expiring soonest (to avoid letting others
        lapse unused). Raises ValueError if none are available or the
        daily redemption cap is hit. Also pushes the rebill date out
        REBILL_INTERVAL_DAYS from `on_date`."""
        used_today = self.redemptions_today.get(on_date, 0)
        if used_today >= MAX_REDEMPTIONS_PER_DAY:
            raise ValueError(
                f"already redeemed {used_today} reset credit(s) on {on_date} "
                f"(limit {MAX_REDEMPTIONS_PER_DAY}/day)"
            )
        available = self.available_credits(on_date)
        if not available:
            raise ValueError(f"no reset credits available on {on_date}")
        credit = min(available, key=lambda c: (c.expires_on or date.max))
        credit.used = True
        credit.used_on = on_date
        self.redemptions_today[on_date] = used_today + 1
        self.rebill_date = on_date + timedelta(days=REBILL_INTERVAL_DAYS)
        return credit

    def expire_stale_credits(self, as_of: date) -> int:
        """No-op cleanup helper -- credits self-report expiry via
        is_available(); this just counts how many are newly expired-and-
        unused as of `as_of`, for reporting."""
        return sum(
            1 for c in self.credits
            if not c.used and c.expires_on is not None and as_of >= c.expires_on
        )
