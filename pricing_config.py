"""Configuration dataclasses and YAML loader for the prop-firm valuation engine."""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Optional

import yaml

TRADING_DAYS_PER_YEAR = 252
BREACH_ACTIONS = ("immediate_reset", "cancel_and_reopen", "wait_for_rebill")
EVAL_MODES = ("no_edge", "positive_drift")


@dataclass(frozen=True)
class FeePath:
    name: str
    monthly_cost: float
    activation_fee: float
    reset_fee: float
    intraday_dll: Optional[float] = None

    def __post_init__(self):
        for attr in ("monthly_cost", "activation_fee", "reset_fee"):
            if getattr(self, attr) < 0:
                raise ValueError(f"{self.name}.{attr} must be >= 0")
        if self.intraday_dll is not None and self.intraday_dll >= 0:
            raise ValueError(f"{self.name}.intraday_dll must be negative (a loss limit)")


@dataclass(frozen=True)
class CombineRules:
    profit_target: float = 9000.0
    trailing_mll: float = 4500.0
    consistency_cap_ratio: float = 0.50
    max_days: int = 1500

    def __post_init__(self):
        if self.profit_target <= 0 or self.trailing_mll <= 0:
            raise ValueError("profit_target and trailing_mll must be positive")
        if not 0 < self.consistency_cap_ratio <= 1:
            raise ValueError("consistency_cap_ratio must be in (0, 1]")


@dataclass(frozen=True)
class Friction:
    round_turn_commission: float = 2.50
    slippage_per_rt: float = 1.00
    trades_per_day: int = 2

    @property
    def daily_cost(self) -> float:
        return (self.round_turn_commission + self.slippage_per_rt) * self.trades_per_day


@dataclass(frozen=True)
class MarketProcess:
    eval_mode: str = "no_edge"
    sharpe_annual: float = 0.0
    sigma_daily: float = 600.0

    def __post_init__(self):
        if self.eval_mode not in EVAL_MODES:
            raise ValueError(f"eval_mode must be one of {EVAL_MODES}")
        if self.sigma_daily <= 0:
            raise ValueError("sigma_daily must be positive")

    @property
    def effective_sharpe(self) -> float:
        return 0.0 if self.eval_mode == "no_edge" else self.sharpe_annual

    @property
    def mu_daily_gross(self) -> float:
        """Gross daily drift implied by the annualised Sharpe: mu = SR / sqrt(252) * sigma."""
        return self.effective_sharpe / math.sqrt(TRADING_DAYS_PER_YEAR) * self.sigma_daily


@dataclass(frozen=True)
class XFAModel:
    horizon_days: int = 252
    payout_interval_days: int = 21
    payout_fraction: float = 0.50
    payout_buffer: float = 0.0
    max_payout: float = 5000.0


@dataclass(frozen=True)
class FarmPolicy:
    breach_action: str = "cancel_and_reopen"
    n_simulations: int = 10000
    max_concurrent_funded: int = 5
    seed: int = 42

    def __post_init__(self):
        if self.breach_action not in BREACH_ACTIONS:
            raise ValueError(f"breach_action must be one of {BREACH_ACTIONS}")


@dataclass(frozen=True)
class EngineConfig:
    firm: str
    billing_cycle_days: int
    profit_split: float
    combine_rules: CombineRules
    paths: Dict[str, FeePath]
    friction: Friction
    market_process: MarketProcess
    xfa: XFAModel
    farm_policy: FarmPolicy = field(default_factory=FarmPolicy)

    def with_market(self, **kwargs) -> "EngineConfig":
        return replace(self, market_process=replace(self.market_process, **kwargs))

    def with_policy(self, **kwargs) -> "EngineConfig":
        return replace(self, farm_policy=replace(self.farm_policy, **kwargs))


def load_config(path: str | Path = Path(__file__).with_name("config.yaml")) -> EngineConfig:
    raw = yaml.safe_load(Path(path).read_text())
    paths = {name: FeePath(name=name, **p) for name, p in raw["paths"].items()}
    return EngineConfig(
        firm=raw["firm"],
        billing_cycle_days=int(raw["billing_cycle_days"]),
        profit_split=float(raw["profit_split"]),
        combine_rules=CombineRules(**raw["combine_rules"]),
        paths=paths,
        friction=Friction(**raw["friction"]),
        market_process=MarketProcess(**raw["market_process"]),
        xfa=XFAModel(**raw.get("xfa", {})),
        farm_policy=FarmPolicy(**raw.get("farm_policy", {})),
    )
