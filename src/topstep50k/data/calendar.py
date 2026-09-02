"""Economic calendar — FOMC and NFP dates for the project data range.

All dates are in US/Eastern calendar days. The calendar is hardcoded
from public Fed / BLS schedules and does NOT use any market data,
so there is ZERO look-ahead risk when using these as trading gates.

FOMC dates are the ANNOUNCEMENT DAY (second day of the 2-day meeting,
when the statement is released at 14:00 ET). The PRE-FOMC EVE is the
calendar day before the announcement.

NFP dates are the BLS Employment Situation release date (typically the
first Friday of the month). These are approximate for 2025-2026 and
should be verified against the official BLS release calendar.

Usage
─────
from topstep50k.data.calendar import (
    is_fomc_day,
    is_fomc_eve,
    is_nfp_day,
    is_high_impact_day,
)

# Use as a daily_filter for any strategy:
strategy = GapFill(..., daily_filter=lambda d: not is_high_impact_day(d))
"""

from __future__ import annotations

from datetime import date, timedelta


# ---------------------------------------------------------------------------
# FOMC announcement dates 2021-2026
# Source: federalreserve.gov/monetarypolicy/fomccalendars.htm
# ---------------------------------------------------------------------------
_FOMC_ANNOUNCEMENT_DATES: frozenset[date] = frozenset([
    # 2021
    date(2021, 1, 27), date(2021, 3, 17), date(2021, 4, 28), date(2021, 6, 16),
    date(2021, 7, 28), date(2021, 9, 22), date(2021, 11, 3), date(2021, 12, 15),
    # 2022
    date(2022, 1, 26), date(2022, 3, 16), date(2022, 5, 4),  date(2022, 6, 15),
    date(2022, 7, 27), date(2022, 9, 21), date(2022, 11, 2), date(2022, 12, 14),
    # 2023
    date(2023, 2, 1),  date(2023, 3, 22), date(2023, 5, 3),  date(2023, 6, 14),
    date(2023, 7, 26), date(2023, 9, 20), date(2023, 11, 1), date(2023, 12, 13),
    # 2024
    date(2024, 1, 31), date(2024, 3, 20), date(2024, 5, 1),  date(2024, 6, 12),
    date(2024, 7, 31), date(2024, 9, 18), date(2024, 11, 7), date(2024, 12, 18),
    # 2025
    date(2025, 1, 29), date(2025, 3, 19), date(2025, 5, 7),  date(2025, 6, 18),
    date(2025, 7, 30), date(2025, 9, 17), date(2025, 10, 29),date(2025, 12, 10),
    # 2026
    date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29),
])

# One calendar day before each FOMC announcement (Lucca-Moench pre-FOMC eve)
_FOMC_EVE_DATES: frozenset[date] = frozenset(
    d - timedelta(days=1) for d in _FOMC_ANNOUNCEMENT_DATES
)

# ---------------------------------------------------------------------------
# NFP release dates 2021-2026
# Source: bls.gov/schedule/news_release/empsit.htm
# Approximately "first Friday of each month" (verified against BLS schedule)
# ---------------------------------------------------------------------------
_NFP_DATES: frozenset[date] = frozenset([
    # 2021
    date(2021, 1, 8),  date(2021, 2, 5),  date(2021, 3, 5),  date(2021, 4, 2),
    date(2021, 5, 7),  date(2021, 6, 4),  date(2021, 7, 2),  date(2021, 8, 6),
    date(2021, 9, 3),  date(2021, 10, 8), date(2021, 11, 5), date(2021, 12, 3),
    # 2022
    date(2022, 1, 7),  date(2022, 2, 4),  date(2022, 3, 4),  date(2022, 4, 1),
    date(2022, 5, 6),  date(2022, 6, 3),  date(2022, 7, 8),  date(2022, 8, 5),
    date(2022, 9, 2),  date(2022, 10, 7), date(2022, 11, 4), date(2022, 12, 2),
    # 2023
    date(2023, 1, 6),  date(2023, 2, 3),  date(2023, 3, 10), date(2023, 4, 7),
    date(2023, 5, 5),  date(2023, 6, 2),  date(2023, 7, 7),  date(2023, 8, 4),
    date(2023, 9, 1),  date(2023, 10, 6), date(2023, 11, 3), date(2023, 12, 8),
    # 2024
    date(2024, 1, 5),  date(2024, 2, 2),  date(2024, 3, 8),  date(2024, 4, 5),
    date(2024, 5, 3),  date(2024, 6, 7),  date(2024, 7, 5),  date(2024, 8, 2),
    date(2024, 9, 6),  date(2024, 10, 4), date(2024, 11, 1), date(2024, 12, 6),
    # 2025
    date(2025, 1, 10), date(2025, 2, 7),  date(2025, 3, 7),  date(2025, 4, 4),
    date(2025, 5, 2),  date(2025, 6, 6),  date(2025, 7, 3),  date(2025, 8, 1),
    date(2025, 9, 5),  date(2025, 10, 3), date(2025, 11, 7), date(2025, 12, 5),
    # 2026
    date(2026, 1, 9),  date(2026, 2, 6),  date(2026, 3, 6),  date(2026, 4, 3),
])


def is_fomc_day(d: date) -> bool:
    """True if d is a FOMC announcement day (statement at 14:00 ET)."""
    return d in _FOMC_ANNOUNCEMENT_DATES


def is_fomc_eve(d: date) -> bool:
    """True if d is the calendar day before a FOMC announcement.

    Lucca & Moench (2015, J. Finance): documented 49bp average excess
    return on the S&P 500 in the 24h before FOMC announcements.
    """
    return d in _FOMC_EVE_DATES


def is_nfp_day(d: date) -> bool:
    """True if d is a BLS Employment Situation release date."""
    return d in _NFP_DATES


def is_high_impact_day(d: date) -> bool:
    """True if d is either a FOMC announcement day OR an NFP day.

    Use as a filter to SKIP strategies sensitive to sharp directional
    moves (e.g. MeanRev, GapFill). Strategies with trend-following bias
    (ORB, IntraDay Momentum) may PREFER these days.
    """
    return is_fomc_day(d) or is_nfp_day(d)


def fomc_announcement_dates() -> frozenset[date]:
    return _FOMC_ANNOUNCEMENT_DATES


def fomc_eve_dates() -> frozenset[date]:
    return _FOMC_EVE_DATES


def nfp_dates() -> frozenset[date]:
    return _NFP_DATES
