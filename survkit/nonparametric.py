"""Kaplan Meier and Nelson Aalen estimators.

Both estimators are computed on the grid of distinct observed times. Delayed
entry (left truncation) is handled through the risk set definition: a subject
is at risk at time u when entry < u <= duration.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._stats import norm_ppf
from ._utils import at_risk_counts, check_survival_data

__all__ = ["KaplanMeier", "NelsonAalen", "EventTable", "event_table"]


@dataclass(frozen=True)
class EventTable:
    """Counts at each distinct observed time."""

    time: np.ndarray
    at_risk: np.ndarray
    events: np.ndarray
    censored: np.ndarray

    def __len__(self) -> int:
        return self.time.size


def event_table(durations, events=None, entry=None) -> EventTable:
    t, e, s = check_survival_data(durations, events, entry)
    times, inverse = np.unique(t, return_inverse=True)
    d = np.bincount(inverse, weights=e.astype(float), minlength=times.size)
    total = np.bincount(inverse, minlength=times.size).astype(float)
    n = at_risk_counts(times, t, s).astype(float)
    return EventTable(times, n, d, total - d)


def _step_lookup(grid, values, at, before_first, left_limit=False):
    """Evaluate a right continuous step function defined on ``grid``."""
    at = np.asarray(at, dtype=float)
    side = "left" if left_limit else "right"
    idx = np.searchsorted(grid, at, side=side) - 1
    out = np.where(idx >= 0, values[np.clip(idx, 0, None)], before_first)
    return out if out.ndim else float(out)


class KaplanMeier:
    """Product limit estimator of the survival function.

    Variance uses Greenwood's formula. Pointwise confidence intervals use the
    log(-log S) transform, which keeps the band inside [0, 1] and has much
    better small sample coverage than the plain linear interval.

    Parameters
    ----------
    alpha : float
        1 - confidence level for the pointwise bands.
    """

    def __init__(self, alpha: float = 0.05):
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")
        self.alpha = alpha

    def fit(self, durations, events=None, entry=None) -> "KaplanMeier":
        table = event_table(durations, events, entry)
        n, d = table.at_risk, table.events
        if np.any((n <= 0) & (d > 0)):
            raise ValueError("an event occurs with an empty risk set; check entry times")
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = np.where(n > 0, 1.0 - d / n, 1.0)
            survival = np.cumprod(factor)
            green_terms = np.where(d > 0, d / (n * (n - d)), 0.0)
        greenwood = np.cumsum(green_terms)

        z = norm_ppf(1 - self.alpha / 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_s = np.log(survival)
            se_loglog = np.sqrt(greenwood) / np.abs(log_s)
            lower = survival ** np.exp(z * se_loglog)
            upper = survival ** np.exp(-z * se_loglog)
        undefined = (survival >= 1.0) | (survival <= 0.0) | ~np.isfinite(se_loglog)
        lower = np.where(undefined, survival, lower)
        upper = np.where(undefined, survival, upper)
        self.table = table
        self.timeline = table.time
        self.survival = survival
        with np.errstate(invalid="ignore"):
            self.variance = np.where(survival > 0, survival ** 2 * greenwood, 0.0)
        self.ci_lower = lower
        self.ci_upper = upper
        return self

    def _check(self):
        if not hasattr(self, "survival"):
            raise RuntimeError("call fit() first")

    def survival_at(self, t, left_limit: bool = False):
        """S(t), or S(t-) when ``left_limit`` is true."""
        self._check()
        return _step_lookup(self.timeline, self.survival, t, 1.0, left_limit)

    def confidence_interval_at(self, t):
        self._check()
        return (_step_lookup(self.timeline, self.ci_lower, t, 1.0),
                _step_lookup(self.timeline, self.ci_upper, t, 1.0))

    def _first_time_at_or_below(self, curve, q):
        hit = np.nonzero(curve <= q + 1e-12)[0]
        return float(self.timeline[hit[0]]) if hit.size else float("inf")

    def percentile(self, q: float = 0.5) -> float:
        """Smallest time at which the survival estimate is <= q."""
        self._check()
        return self._first_time_at_or_below(self.survival, q)

    @property
    def median(self) -> float:
        return self.percentile(0.5)

    def median_confidence_interval(self):
        """Brookmeyer and Crowley style interval, read off the pointwise bands."""
        self._check()
        return (self._first_time_at_or_below(self.ci_lower, 0.5),
                self._first_time_at_or_below(self.ci_upper, 0.5))

    def restricted_mean(self, tau: float):
        """Restricted mean survival time up to ``tau`` and its standard error.

        RMST is the area under the step curve on [0, tau]. The variance is the
        standard Greenwood type delta method estimate.
        """
        self._check()
        grid = self.timeline[self.timeline < tau]
        knots = np.concatenate([[0.0], grid, [tau]])
        knots = np.unique(knots)
        s_left = self.survival_at(knots[:-1])
        widths = np.diff(knots)
        rmst = float(np.sum(np.atleast_1d(s_left) * widths))

        tab = self.table
        mask = (tab.time <= tau) & (tab.events > 0) & (tab.at_risk > tab.events)
        var = 0.0
        for t_j, n_j, d_j in zip(tab.time[mask], tab.at_risk[mask], tab.events[mask]):
            sel = knots[:-1] >= t_j
            area_after = float(np.sum(np.atleast_1d(s_left)[sel] * widths[sel]))
            var += area_after ** 2 * d_j / (n_j * (n_j - d_j))
        return rmst, float(np.sqrt(var))

    def summary(self, events_only: bool = True) -> str:
        self._check()
        tab = self.table
        rows = ["{:>10} {:>8} {:>7} {:>9} {:>9} {:>9} {:>9}".format(
            "time", "at_risk", "events", "survival", "std_err", "lower", "upper")]
        for i in range(len(tab)):
            if events_only and tab.events[i] == 0:
                continue
            rows.append("{:>10.4g} {:>8d} {:>7d} {:>9.4f} {:>9.4f} {:>9.4f} {:>9.4f}".format(
                tab.time[i], int(tab.at_risk[i]), int(tab.events[i]), self.survival[i],
                np.sqrt(self.variance[i]) if np.isfinite(self.variance[i]) else float("nan"),
                self.ci_lower[i], self.ci_upper[i]))
        return "\n".join(rows)


class NelsonAalen:
    """Nelson Aalen estimator of the cumulative hazard H(t) = sum d / n.

    Variance uses the Aalen estimator sum d / n^2, and the confidence interval
    is built on the log scale.
    """

    def __init__(self, alpha: float = 0.05):
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")
        self.alpha = alpha

    def fit(self, durations, events=None, entry=None) -> "NelsonAalen":
        table = event_table(durations, events, entry)
        n, d = table.at_risk, table.events
        with np.errstate(divide="ignore", invalid="ignore"):
            increments = np.where(d > 0, d / n, 0.0)
            var_inc = np.where(d > 0, d / n ** 2, 0.0)
        H = np.cumsum(increments)
        var = np.cumsum(var_inc)
        z = norm_ppf(1 - self.alpha / 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            mult = np.exp(z * np.sqrt(var) / H)
        mult = np.where(H > 0, mult, 1.0)
        self.table = table
        self.timeline = table.time
        self.cumulative_hazard = H
        self.variance = var
        self.ci_lower = H / mult
        self.ci_upper = H * mult
        return self

    def cumulative_hazard_at(self, t, left_limit: bool = False):
        if not hasattr(self, "cumulative_hazard"):
            raise RuntimeError("call fit() first")
        return _step_lookup(self.timeline, self.cumulative_hazard, t, 0.0, left_limit)
