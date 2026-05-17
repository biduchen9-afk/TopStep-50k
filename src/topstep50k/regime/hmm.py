"""Hidden Markov Model regime filter, audit-safe.

Design contract -- read this before changing anything.

  1. **Fit on the training window only.** `fit_hmm()` takes a feature
     matrix and returns a fitted model. The CALLER is responsible for
     ensuring those features come from the walk-forward TRAIN fold only.
     There is no API on this module that pulls features by date range;
     the boundary check is the caller's.

  2. **Online inference is FORWARD-FILTER, not Viterbi smoothing.**
     `HMMRegimeFilter.update(features_t)` runs ONE step of the forward
     pass and returns the posterior P(state | obs_1..t). It does NOT
     re-fit, does NOT smooth backward, and never reads features_s for
     s > t. Math: alpha_t(j) = (sum_i alpha_{t-1}(i) * A[i,j]) * B_j(o_t),
     then normalise. [ref:rabiner_1989] section II.

  3. **Audit on every transition.** When the most-likely state changes
     across two consecutive bars, an audit event of kind
     `regime_transition` is emitted with the prior and posterior
     distributions. Strategies should NEVER use the regime label to act
     on bar t; they read it AFTER on_bar(t) and act on bar t+1 (the
     engine's deferred-fill contract already enforces this).

References:
  Hamilton 1989 [ref:hamilton_1989] foundational regime-switching
  Nystrup et al. 2017 [ref:nystrup_2017] HMM on financial returns
  Rabiner 1989 [ref:rabiner_1989] forward-filter recurrence
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

import numpy as np

from topstep50k.audit import AuditEvent, AuditLog
from topstep50k.engine.clock import Clock


@dataclass(frozen=True)
class HMMRegimeResult:
    """The fitted artefact. State indices are arbitrary (they come from
    EM); use `state_labels` for human-readable mapping if needed.
    """

    n_states: int
    transition: np.ndarray  # (k, k)
    means: np.ndarray       # (k, d)
    covars: np.ndarray      # (k, d, d) full or (k, d) diag depending on covariance_type
    start_prob: np.ndarray  # (k,)
    covariance_type: str
    train_log_likelihood: float
    n_features: int
    converged: bool
    n_iter_used: int


@dataclass(frozen=True)
class RegimeAssignment:
    ts: datetime
    most_likely_state: int
    posterior: np.ndarray  # (k,)


def fit_hmm(
    features: np.ndarray,
    *,
    n_states: int = 3,
    covariance_type: str = "full",
    n_iter: int = 200,
    tol: float = 1e-4,
    seed: int | None = 0,
) -> HMMRegimeResult:
    """Fit a Gaussian HMM via EM (Baum-Welch).

    Parameters
    ----------
    features : shape (T_train, d). Train-fold features ONLY.
    n_states : 2-5 is typical for financial regimes.
    covariance_type : 'full', 'diag', 'tied', 'spherical' (passed through to hmmlearn).
    seed : random_state for reproducibility.

    Returns a frozen result containing the EM-learned parameters; the
    online filter constructed from it does not retrain.
    """
    from hmmlearn.hmm import GaussianHMM

    if features.ndim != 2:
        raise ValueError("features must be 2D (T, d)")
    if len(features) < n_states * 10:
        raise ValueError(
            f"fit needs at least {n_states * 10} samples for {n_states} states; "
            f"got {len(features)}"
        )
    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        tol=tol,
        random_state=seed,
    )
    model.fit(features)
    return HMMRegimeResult(
        n_states=n_states,
        transition=model.transmat_.copy(),
        means=model.means_.copy(),
        covars=model.covars_.copy(),
        start_prob=model.startprob_.copy(),
        covariance_type=covariance_type,
        train_log_likelihood=float(model.score(features)),
        n_features=features.shape[1],
        converged=bool(model.monitor_.converged),
        n_iter_used=int(model.monitor_.iter),
    )


class HMMRegimeFilter:
    """Online forward-filter wrapper around a fitted HMM.

    Usage:
        result = fit_hmm(train_features)            # train fold only
        flt = HMMRegimeFilter(result)
        for ts, x in test_features_stream:          # bar by bar
            assignment = flt.update(ts, x, clock=clock, audit=audit)
            # use assignment to gate trading decisions ON bar t+1

    Importantly: features for ts must already be `clock.now()`-visible
    (e.g. computed from bars whose ts is <= now). The filter calls
    `clock.assert_visible(ts)` defensively.
    """

    def __init__(
        self,
        fitted: HMMRegimeResult,
        *,
        symbol_tag: str = "",
    ) -> None:
        self._fit = fitted
        self._symbol_tag = symbol_tag
        self._alpha: np.ndarray | None = None  # current forward message
        self._last_state: int | None = None
        # Pre-compute multivariate-normal log-pdf normalisers for speed
        self._k = fitted.n_states
        self._d = fitted.n_features
        self._inv_covars: list[np.ndarray] = []
        self._logdets: list[float] = []
        for j in range(self._k):
            cov = self._extract_cov(j)
            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0:
                # Regularise
                cov = cov + 1e-9 * np.eye(self._d)
                _, logdet = np.linalg.slogdet(cov)
            self._inv_covars.append(np.linalg.inv(cov))
            self._logdets.append(float(logdet))
        # Pre-compute log(transition) for numerically-stable updates
        with np.errstate(divide="ignore"):
            self._log_A = np.where(
                fitted.transition > 0,
                np.log(np.maximum(fitted.transition, 1e-300)),
                -1e300,
            )
            self._log_pi = np.where(
                fitted.start_prob > 0,
                np.log(np.maximum(fitted.start_prob, 1e-300)),
                -1e300,
            )

    def _extract_cov(self, j: int) -> np.ndarray:
        c = self._fit.covars
        if c.ndim == 3:
            return c[j]
        if c.ndim == 2:  # diag
            return np.diag(c[j])
        if c.ndim == 1:  # spherical
            return c[j] * np.eye(self._d)
        raise ValueError(f"unexpected covars shape {c.shape}")

    def _log_emit(self, j: int, x: np.ndarray) -> float:
        diff = x - self._fit.means[j]
        mahal = float(diff @ self._inv_covars[j] @ diff)
        return -0.5 * (self._d * np.log(2 * np.pi) + self._logdets[j] + mahal)

    def update(
        self,
        ts: datetime,
        features_t: np.ndarray,
        *,
        clock: Clock | None = None,
        audit: AuditLog | None = None,
    ) -> RegimeAssignment:
        """One forward-filter step. Returns posterior over states after
        observing features_t.

        Log-domain recurrence:
            log_alpha_t(j) = logsumexp_i [log_alpha_{t-1}(i) + log_A[i,j]] + log_b_j(x_t)
        Normalised at the end so it stays interpretable as a posterior.
        """
        if clock is not None:
            clock.assert_visible(ts, "HMM regime feature")
        if features_t.shape != (self._d,):
            raise ValueError(
                f"expected feature vector of length {self._d}, got {features_t.shape}"
            )

        log_b = np.array([self._log_emit(j, features_t) for j in range(self._k)])

        if self._alpha is None:
            log_alpha = self._log_pi + log_b
        else:
            prev_log = np.log(np.maximum(self._alpha, 1e-300))
            # log_alpha_new[j] = logsumexp_i (prev_log[i] + log_A[i,j]) + log_b[j]
            mat = prev_log[:, None] + self._log_A  # (k, k); axis 0 = prev state i
            max_per_col = mat.max(axis=0)
            stable = np.log(np.exp(mat - max_per_col).sum(axis=0)) + max_per_col
            log_alpha = stable + log_b

        # Normalise to a posterior
        m = log_alpha.max()
        alpha = np.exp(log_alpha - m)
        alpha = alpha / alpha.sum()
        self._alpha = alpha
        most_likely = int(alpha.argmax())

        if audit is not None and self._last_state is not None and most_likely != self._last_state:
            audit.record(AuditEvent(
                ts=ts, kind="regime_transition",
                payload={
                    "tag": self._symbol_tag,
                    "from": self._last_state,
                    "to": most_likely,
                    "posterior": [float(p) for p in alpha],
                },
            ))
        self._last_state = most_likely
        return RegimeAssignment(ts=ts, most_likely_state=most_likely, posterior=alpha)

    @property
    def posterior(self) -> np.ndarray | None:
        return None if self._alpha is None else self._alpha.copy()
