import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, pandas as pd
from pricing_config import load_config
from real_engine import rules_from_config, Stream, FundedPlan
from farm_real import FarmSpec, run_farm
from contract_eval import CombineSizing, contract_metrics
from portfolio import DEV_END, basket, make_sleeve
from run_farm_dev import plan, SZ

PATH = "no_activation_dll"
cfg = load_config()
rules = rules_from_config(cfg, PATH)
pd.set_option("display.width", 250, "display.max_columns", 40)


def real_stream():
    return basket([make_sleeve("nq", "noise_area", {}, name="nq_noise")], name="nq")


COST_DAY = 0.87 * (0.25 * 20 + 2.5) / 10   # trades/day * round-trip cost per micro


def gauss_stream(real, seed, c_sd, mean_per_atr=0.0, cost=0.0, scale=1.0):
    ok = real.ok
    rng = np.random.default_rng(seed)
    n = len(real.dates)
    sig = np.where(ok, c_sd * scale * real.atr_usd, 0.0)
    pnl = rng.normal(0, 1, n) * sig + np.where(ok, mean_per_atr * real.atr_usd, 0.0) - np.where(ok, cost, 0.0)
    U = rng.uniform(1e-12, 1, n)
    low = (pnl - np.sqrt(pnl ** 2 - 2 * sig ** 2 * np.log(U))) / 2     # Brownian-bridge minimum
    low = np.minimum(low, np.minimum(pnl, 0))
    return Stream("gauss", real.dates, pnl, low, real.atr_usd, ok)
