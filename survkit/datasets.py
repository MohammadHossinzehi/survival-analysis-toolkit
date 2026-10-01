"""Bundled data and simulators.

``load_gehan`` is the classic 6MP leukaemia remission trial (Freireich et al.
1963, analysed by Gehan 1965): 42 children, half on 6 mercaptopurine and half
on placebo, remission time in weeks. It is small enough to check by hand and
every survival textbook reports reference numbers for it, which makes it the
regression test fixture for this package.
"""
from __future__ import annotations

import numpy as np

__all__ = ["load_gehan", "simulate_weibull_ph"]

_SIX_MP = [(6, 1), (6, 1), (6, 1), (6, 0), (7, 1), (9, 0), (10, 1), (10, 0), (11, 0),
           (13, 1), (16, 1), (17, 0), (19, 0), (20, 0), (22, 1), (23, 1), (25, 0),
           (32, 0), (32, 0), (34, 0), (35, 0)]
_PLACEBO = [1, 1, 2, 2, 3, 4, 4, 5, 5, 8, 8, 8, 8, 11, 11, 12, 12, 15, 17, 22, 23]


def load_gehan():
    """Return (weeks, relapsed, placebo) as NumPy arrays.

    ``relapsed`` is 1 for an observed relapse and 0 for a censored remission;
    ``placebo`` is 1 for the placebo arm and 0 for 6MP.
    """
    weeks = np.array([w for w, _ in _SIX_MP] + _PLACEBO, dtype=float)
    relapsed = np.array([s for _, s in _SIX_MP] + [1] * len(_PLACEBO), dtype=int)
    placebo = np.array([0] * len(_SIX_MP) + [1] * len(_PLACEBO), dtype=int)
    return weeks, relapsed, placebo


def simulate_weibull_ph(n, beta, shape=1.5, scale=10.0, censor_rate=0.05, X=None,
                        admin_censor=None, rng=None):
    """Draw from a Weibull proportional hazards model.

    H(t | x) = (t / scale) ** shape * exp(x'beta), so inverse transform
    sampling gives T = scale * (-log U / exp(x'beta)) ** (1 / shape).
    Independent exponential censoring with rate ``censor_rate`` (0 disables)
    and an optional administrative cutoff are applied.

    Returns (X, durations, events, true_event_times).
    """
    rng = np.random.default_rng(rng)
    beta = np.atleast_1d(np.asarray(beta, float))
    if X is None:
        X = rng.standard_normal((n, beta.size))
    X = np.asarray(X, float)
    u = rng.random(n)
    T = scale * (-np.log(u) / np.exp(X @ beta)) ** (1.0 / shape)
    C = rng.exponential(1.0 / censor_rate, n) if censor_rate > 0 else np.full(n, np.inf)
    if admin_censor is not None:
        C = np.minimum(C, admin_censor)
    durations = np.minimum(T, C)
    events = (T <= C).astype(int)
    return X, durations, events, T
