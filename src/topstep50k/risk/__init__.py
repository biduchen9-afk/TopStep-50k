"""Risk management.

Only one module right now: `sizing` -- an EV-gated fractional-Kelly
position-size multiplier that ACTIVATES only when the strategy's OOS
performance on the previous walk-forward fold clears statistical gates
(Deflated Sharpe and bootstrap pass-rate). Until those gates pass, the
recommended size is zero -- "don't trade what isn't proven."
"""

from topstep50k.risk.sizing import (
    EVGate,
    EVGateResult,
    SizingDecision,
    fractional_kelly,
)

__all__ = ["EVGate", "EVGateResult", "SizingDecision", "fractional_kelly"]
