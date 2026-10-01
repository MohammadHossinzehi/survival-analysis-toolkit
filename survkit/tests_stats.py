"""K sample tests for equality of survival curves.

All tests share one engine: at each distinct event time the observed and
expected event counts per group are accumulated with a time dependent weight,
along with the hypergeometric covariance. The weight picks the test:

=====================  =========================================  ==========
name                   weight w(t)                                emphasis
=====================  =========================================  ==========
``logrank``            1                                          late
``wilcoxon``           n(t)  (Gehan and Breslow)                      early
``tarone_ware``        sqrt(n(t))                                 middle
``peto``               S~(t) Peto and Peto modified survival          early
``fleming_harrington`` S(t-)^p (1 - S(t-))^q, pooled Kaplan Meier  tunable
=====================  =========================================  ==========
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._stats import chi2_sf, norm_sf
from ._utils import at_risk_counts, check_survival_data

__all__ = ["logrank_test", "logrank_trend_test", "TestResult"]


@dataclass(frozen=True)
class TestResult:
    statistic: float
    df: int
    p_value: float
    observed: np.ndarray
    expected: np.ndarray
    groups: np.ndarray
    weighting: str

    def __str__(self) -> str:
        lines = [f"{self.weighting} test: chi2 = {self.statistic:.4f} on {self.df} df, "
                 f"p = {self.p_value:.4g}",
                 "{:>12} {:>10} {:>10}".format("group", "observed", "expected")]
        for g, o, e in zip(self.groups, self.observed, self.expected):
            lines.append("{:>12} {:>10.2f} {:>10.2f}".format(str(g), o, e))
        return "\n".join(lines)


def _weights(name, n, d, p, q):
    name = name.lower()
    if name == "logrank":
        return np.ones_like(n)
    if name in ("wilcoxon", "gehan", "breslow"):
        return n.copy()
    if name == "tarone_ware":
        return np.sqrt(n)
    if name == "peto":
        return np.cumprod(1.0 - d / (n + 1.0))
    if name == "fleming_harrington":
        s = np.cumprod(1.0 - d / n)
        s_left = np.concatenate([[1.0], s[:-1]])
        return s_left ** p * (1.0 - s_left) ** q
    raise ValueError(f"unknown weighting {name!r}")


def _score_components(durations, events, groups, entry, weighting, p, q):
    t, e, s = check_survival_data(durations, events, entry)
    g = np.asarray(groups).ravel()
    if g.shape != t.shape:
        raise ValueError("groups must have the same length as durations")
    labels = np.unique(g)
    if labels.size < 2:
        raise ValueError("need at least two groups")
    k = labels.size

    times = np.unique(t[e])
    n_g = np.empty((times.size, k))
    d_g = np.empty((times.size, k))
    for j, lab in enumerate(labels):
        m = g == lab
        n_g[:, j] = at_risk_counts(times, t[m], s[m])
        tm = np.sort(t[m & e])
        d_g[:, j] = np.searchsorted(tm, times, side="right") - \
            np.searchsorted(tm, times, side="left")
    n = n_g.sum(axis=1)
    d = d_g.sum(axis=1)
    w = _weights(weighting, n, d, p, q)

    expected_g = d[:, None] * n_g / n[:, None]
    U = (w[:, None] * (d_g - expected_g)).sum(axis=0)

    frac = n_g / n[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        hyper = np.where(n > 1, d * (n - d) / (n - 1), 0.0)
    scale = w ** 2 * hyper
    V = np.einsum("t,ti->i", scale, frac)[:, None] * np.eye(k) \
        - np.einsum("t,ti,tj->ij", scale, frac, frac)
    return labels, U, V, d_g.sum(axis=0), expected_g.sum(axis=0)


def logrank_test(durations, events=None, groups=None, entry=None,
                 weighting: str = "logrank", p: float = 0.0, q: float = 0.0) -> TestResult:
    """Weighted K sample log rank family test.

    Returns a chi square statistic on K - 1 degrees of freedom. The covariance
    of the score vector is singular (its rows sum to zero), so the last group
    is dropped before solving, the standard construction.
    """
    if groups is None:
        raise ValueError("groups is required")
    labels, U, V, obs, exp = _score_components(durations, events, groups, entry, weighting, p, q)
    k = labels.size
    U_r, V_r = U[: k - 1], V[: k - 1, : k - 1]
    stat = float(U_r @ np.linalg.solve(V_r, U_r))
    name = weighting if weighting != "fleming_harrington" else f"fleming_harrington(p={p}, q={q})"
    return TestResult(stat, k - 1, chi2_sf(stat, k - 1), obs, exp, labels, name)


def logrank_trend_test(durations, events=None, groups=None, scores=None, entry=None):
    """Log rank test for trend across ordered groups.

    ``scores`` assigns a numeric dose to each sorted group label (defaults to
    0, 1, 2, ...). The statistic c'U / sqrt(c'Vc) is standard normal under the
    null, and a one degree of freedom test is far more powerful than the
    omnibus K - 1 df test when the effect is monotone.

    Returns (z, two_sided_p).
    """
    if groups is None:
        raise ValueError("groups is required")
    labels, U, V, _, _ = _score_components(durations, events, groups, entry, "logrank", 0, 0)
    c = np.arange(labels.size, dtype=float) if scores is None else np.asarray(scores, float)
    if c.shape != (labels.size,):
        raise ValueError("one score per group is required")
    z = float(c @ U / np.sqrt(c @ V @ c))
    return z, 2 * norm_sf(abs(z))
