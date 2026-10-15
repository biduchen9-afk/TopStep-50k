"""Shared helpers for pre-registrations 12-13: the 'not greedy' acceptance rule with the dev-half stability checks, and the edge-removed placebo streams."""
from __future__ import annotations
import numpy as np
from portfolio import DEV_END
from real_engine import Stream


def halves(nets):
    nets = np.asarray(nets, dtype=float)
    return float(nets[:5].mean()), float(nets[5:].mean())


def accept(base: dict, r: dict, combine: bool = False) -> list:
    """Reasons a cell is rejected (empty list = passes). base / r carry dev_mean, dev_p25, dev_worst, paid_share, nets, days_first_payout (+ fees_per_funded for Combine cells)."""
    why = []
    if not r["dev_mean"] > base["dev_mean"]: why.append("lower mean")
    if not r["dev_p25"] > base["dev_p25"]: why.append("worse p25")
    if not r["dev_worst"] >= (base["dev_worst"] * 1.25 if base["dev_worst"] < 0 else base["dev_worst"] * 0.75): why.append("deeper worst farm")
    if not r["paid_share"] >= base["paid_share"] - 0.05: why.append("fewer accounts paid")
    h0, h1 = halves(base["nets"]), halves(r["nets"])
    if not (h1[0] > h0[0] and h1[1] > h0[1]): why.append("not better in both halves")
    if combine:
        if not r["fees_per_funded"] <= 1.10 * base["fees_per_funded"]: why.append("Combine fees per funded >10% higher")
        if not r["days_first_payout"] <= 1.10 * base["days_first_payout"]: why.append("first payout >10% later")
    return why


def placebo_streams(ST, groups=("noise", "gap")):
    out = {}
    for g in groups:
        s = ST[g]; dv = (s.dates <= DEV_END) & s.ok
        mu_R = float(np.mean(s.pnl[dv] / s.atr_usd[dv])); dm = np.where(s.ok, s.pnl - mu_R * np.nan_to_num(s.atr_usd), s.pnl)
        out[g] = Stream(s.name, s.dates, dm, np.minimum(s.low - (s.pnl - dm), 0.0), s.atr_usd, s.ok)
    return out
