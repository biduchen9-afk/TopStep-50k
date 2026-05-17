from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from topstep50k.engine.ledger import Ledger
from topstep50k.engine.types import Fill, Instrument, OrderSide


ES = Instrument(
    symbol="ES",
    point_value=Decimal("50"),
    tick_size=Decimal("0.25"),
    commission_per_side=Decimal("1.50"),
)


def utc(y, m, d, h=0):
    return datetime(y, m, d, h, tzinfo=timezone.utc)


def fill(side, qty, price, when=utc(2026, 1, 1, 10)):
    return Fill(
        order_ts=when,
        fill_ts=when,
        symbol="ES",
        side=side,
        qty=qty,
        price=price,
        commission=ES.commission_per_side * qty,
    )


def ledger() -> Ledger:
    return Ledger(starting_balance=Decimal("50000"), instruments={"ES": ES})


class TestPositionAccounting:
    def test_open_long(self):
        lg = ledger()
        delta = lg.apply_fill(fill(OrderSide.BUY, 2, 4500.00))
        assert delta == Decimal("0")
        assert lg.positions["ES"].qty == 2
        assert lg.positions["ES"].avg_price == 4500.00

    def test_add_to_long_weighted_avg(self):
        lg = ledger()
        lg.apply_fill(fill(OrderSide.BUY, 2, 4500.00))
        lg.apply_fill(fill(OrderSide.BUY, 2, 4502.00))
        # avg of 4 contracts: (2*4500 + 2*4502)/4 = 4501
        assert lg.positions["ES"].qty == 4
        assert lg.positions["ES"].avg_price == 4501.00

    def test_partial_close_realises(self):
        lg = ledger()
        lg.apply_fill(fill(OrderSide.BUY, 4, 4500.00))
        lg.apply_fill(fill(OrderSide.SELL, 2, 4502.00))
        # closed 2 contracts, +2 points each = +8 ticks * 2 ct * $12.50 = $200
        assert lg.positions["ES"].qty == 2
        assert lg.realised_pnl == Decimal("200")

    def test_flip_long_to_short(self):
        lg = ledger()
        lg.apply_fill(fill(OrderSide.BUY, 2, 4500.00))
        lg.apply_fill(fill(OrderSide.SELL, 5, 4501.00))
        # closes 2 longs at +1pt = +$100; opens 3 shorts at 4501
        assert lg.realised_pnl == Decimal("100")
        assert lg.positions["ES"].qty == -3
        assert lg.positions["ES"].avg_price == 4501.00

    def test_equity_mark_to_market(self):
        lg = ledger()
        lg.apply_fill(fill(OrderSide.BUY, 1, 4500.00))
        eq = lg.equity({"ES": 4502.00})
        # +2 points unrealised = +$100, minus $1.50 commission
        assert eq == Decimal("50000") + Decimal("100") - Decimal("1.50")


class TestDayBoundaries:
    def test_pnl_recorded_per_day(self):
        from datetime import date

        lg = ledger()
        lg.apply_fill(fill(OrderSide.BUY, 1, 4500.00))
        lg.apply_fill(fill(OrderSide.SELL, 1, 4502.00))
        lg.begin_day(date(2026, 1, 2), current_equity=lg.equity({}))
        # closed already so no marks needed
        eod = lg.equity({})
        booked = lg.end_day(date(2026, 1, 2), eod)
        # SOD equity == EOD equity here, so day delta is 0
        assert booked == Decimal("0")
        # But realised PnL from yesterday should still show on the account
        assert lg.realised_pnl == Decimal("100")
