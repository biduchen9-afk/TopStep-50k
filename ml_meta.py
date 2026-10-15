"""Walk-forward meta-labeling (pre-registration 8): numpy-only logistic, L1-logistic and gradient-boosted stumps decide whether to take each trade of a base sleeve.
Training uses only trades from EARLIER calendar years (at least 3 years, retrained every year); only out-of-sample days are scored."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from portfolio import Sleeve

FEATS = ["adx", "er20", "rsi14", "roc20", "dist200", "atr_pct", "hv_pct", "vr5", "hurst", "gapA", "volratio", "dow1", "dow2", "dow3", "dow4", "tom"]


def design(dates: np.ndarray, F: Dict[str, np.ndarray], A: np.ndarray, gap: np.ndarray, volratio: np.ndarray) -> np.ndarray:
    d = pd.DatetimeIndex(dates); n = len(d)
    X = np.full((n, len(FEATS)), np.nan)
    X[:, 0], X[:, 1], X[:, 2] = F["adx"], F["er20"], F["rsi14"]
    X[:, 3] = np.clip(F["roc20"], -0.3, 0.3)
    X[:, 4] = np.clip((F["close"] - F["sma200"]) / A, -30, 30)
    X[:, 5], X[:, 6], X[:, 7], X[:, 8] = F["atr_pct"], F["hv_pct"], F["vr5"], F["hurst"]
    X[:, 9] = np.clip(gap / A, -3, 3); X[:, 10] = volratio
    for k in range(1, 5):
        X[:, 10 + k] = (d.dayofweek.to_numpy() == k).astype(float)
    dm = d.day.to_numpy(); X[:, 15] = ((dm >= 28) | (dm <= 3)).astype(float)
    return X


def _fit_logistic(X, y, l2=1.0, l1=0.0, iters=300):
    n, k = X.shape; w = np.zeros(k + 1); Xb = np.c_[np.ones(n), X]
    lr = 0.5 / (0.25 * (Xb ** 2).sum(0).max() / n + l2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Xb @ w, -30, 30)))
        g = Xb.T @ (p - y) / n + l2 * np.r_[0.0, w[1:]] / n
        w = w - lr * g * 4
        if l1 > 0:
            w[1:] = np.sign(w[1:]) * np.maximum(np.abs(w[1:]) - lr * l1 * 4, 0.0)
    return w


def _pred_logistic(w, X):
    return 1 / (1 + np.exp(-np.clip(np.c_[np.ones(len(X)), X] @ w, -30, 30)))


def _fit_gbm(X, y, rounds=100, lr=0.1):
    n, k = X.shape; p0 = np.clip(y.mean(), 0.02, 0.98); f0 = np.log(p0 / (1 - p0)); F_ = np.full(n, f0); trees = []
    qs = [np.unique(np.quantile(X[:, j], np.linspace(0.1, 0.9, 9))) for j in range(k)]
    for _ in range(rounds):
        p = 1 / (1 + np.exp(-F_)); g = y - p; hs = p * (1 - p) + 1e-6
        best = None
        for j in range(k):
            for th in qs[j]:
                L = X[:, j] <= th
                if L.sum() < 20 or (~L).sum() < 20:
                    continue
                gl, gr = g[L].sum(), g[~L].sum(); hl, hr = hs[L].sum(), hs[~L].sum()
                gain = gl ** 2 / hl + gr ** 2 / hr
                if best is None or gain > best[0]:
                    best = (gain, j, th, gl / hl, gr / hr)
        if best is None:
            break
        _, j, th, vl, vr = best
        trees.append((j, th, lr * vl, lr * vr)); F_ += np.where(X[:, j] <= th, lr * vl, lr * vr)
    return f0, trees


def _pred_gbm(model, X):
    f0, trees = model; F_ = np.full(len(X), f0)
    for j, th, vl, vr in trees:
        F_ += np.where(X[:, j] <= th, vl, vr)
    return 1 / (1 + np.exp(-F_))


def meta_label(base: Sleeve, X: np.ndarray, model: str, thr: float, first_oos_year: int = 2013, min_train: int = 400) -> Sleeve:
    d = pd.DatetimeIndex(base.dates); yrs = d.year.to_numpy(); traded = base.ok & (base.pnl != 0)
    y_all = (base.pnl > 0).astype(float)
    take = np.zeros(len(d), dtype=bool); scored = np.zeros(len(d), dtype=bool)
    for Y in range(first_oos_year, int(yrs.max()) + 1):
        tr_idx = np.flatnonzero(traded & (yrs < Y)); te_idx = np.flatnonzero(base.ok & (yrs == Y))
        if len(tr_idx) < min_train or len(te_idx) == 0:
            continue
        Xt = X[tr_idx]; med = np.nanmedian(Xt, axis=0); med = np.where(np.isfinite(med), med, 0.0)
        fill = lambda Z: np.where(np.isfinite(Z), Z, med)
        Xt = fill(Xt); mu, sd = Xt.mean(0), Xt.std(0) + 1e-9; Z = (Xt - mu) / sd; Zte = (fill(X[te_idx]) - mu) / sd
        if model == "logit":
            P = _pred_logistic(_fit_logistic(Z, y_all[tr_idx], l2=1.0), Zte)
        elif model == "lasso":
            P = _pred_logistic(_fit_logistic(Z, y_all[tr_idx], l2=0.0, l1=0.02), Zte)
        else:
            P = _pred_gbm(_fit_gbm(Z, y_all[tr_idx]), Zte)
        take[te_idx] = P > thr; scored[te_idx] = True
    pnl = np.where(take, base.pnl, 0.0); low = np.where(take, base.low, 0.0)
    return Sleeve(base.name + f"|ml_{model}_{thr}", base.dates, pnl, low, base.atr, base.ok & scored, base.volratio)
