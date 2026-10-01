"""Parametric Weibull and exponential fits for right censored data.

With r observed events the Weibull log likelihood
    l(k, lam) = sum_events [log k - k log lam + (k - 1) log t] - sum_all (t / lam)^k
can be profiled analytically over the scale: lam(k)^k = sum t^k / r. What is
left is a one dimensional, strictly monotone score equation in the shape k,
solved here by safeguarded Newton iteration (bisection fallback), so the fit
cannot wander off the way a naive two parameter Newton step can.
"""
from __future__ import annotations

import math

import numpy as np

from ._stats import chi2_sf, norm_ppf
from ._utils import check_survival_data

__all__ = ["WeibullFitter", "ExponentialFitter"]


class ExponentialFitter:
    """Constant hazard model. MLE rate = events / total exposure."""

    def fit(self, durations, events=None) -> "ExponentialFitter":
        t, e, _ = check_survival_data(durations, events)
        r = int(e.sum())
        exposure = float(t.sum())
        if r == 0 or exposure <= 0:
            raise ValueError("need at least one event and positive exposure")
        self.rate = r / exposure
        self.rate_se = math.sqrt(r) / exposure
        self.log_likelihood = r * math.log(self.rate) - self.rate * exposure
        self.n_events = r
        return self

    def survival_at(self, t):
        return np.exp(-self.rate * np.asarray(t, float))

    @property
    def median(self) -> float:
        return math.log(2) / self.rate


class WeibullFitter:
    """S(t) = exp(-(t / scale) ** shape).

    After fitting, ``shape``, ``scale``, their standard errors (delta method
    from the observed information on the log scale), ``log_likelihood`` and
    ``exponential_lr_test`` (likelihood ratio test of shape == 1) are set.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    @staticmethod
    def _profile_score(k, logt, e, r, mean_log_event):
        # d/dk of the profile log likelihood divided by r; increasing in k.
        a = k * logt
        a -= a.max()
        w = np.exp(a)
        return float((w @ logt) / w.sum() - 1.0 / k - mean_log_event)

    def fit(self, durations, events=None) -> "WeibullFitter":
        t, e, _ = check_survival_data(durations, events)
        if np.any(t <= 0):
            raise ValueError("Weibull fitting needs strictly positive durations")
        r = int(e.sum())
        if r == 0:
            raise ValueError("need at least one event")
        logt = np.log(t)
        mean_log_event = float(logt[e].mean())
        f = lambda k: self._profile_score(k, logt, e, r, mean_log_event)

        lo, hi = 1e-3, 1.0
        while f(hi) < 0:
            lo, hi = hi, hi * 2
            if hi > 1e4:
                raise RuntimeError("shape diverges: all events share one time?")
        while f(lo) > 0:
            lo /= 2
            if lo < 1e-10:
                raise RuntimeError("shape collapses to zero")
        k = 0.5 * (lo + hi)
        for _ in range(200):
            fk = f(k)
            if fk > 0:
                hi = k
            else:
                lo = k
            h = 1e-6 * k
            deriv = (f(k + h) - f(k - h)) / (2 * h)
            k_new = k - fk / deriv if deriv > 0 else 0.5 * (lo + hi)
            if not lo < k_new < hi:
                k_new = 0.5 * (lo + hi)
            if abs(k_new - k) < 1e-13 * k:
                k = k_new
                break
            k = k_new
        tk = t ** k
        scale = (tk.sum() / r) ** (1.0 / k)

        self.shape, self.scale = float(k), float(scale)
        self.log_likelihood = self._loglik(np.log(k), np.log(scale), t, e)
        self._observed_information(t, e)
        expo = ExponentialFitter().fit(t, e)
        lr = 2 * (self.log_likelihood - expo.log_likelihood)
        self.exponential_lr_test = (lr, chi2_sf(max(lr, 0.0), 1))
        self.n_events = r
        return self

    @staticmethod
    def _loglik(log_k, log_lam, t, e):
        k, lam = math.exp(log_k), math.exp(log_lam)
        z = (t / lam) ** k
        return float(np.sum(e * (log_k - k * log_lam + (k - 1) * np.log(t))) - z.sum())

    def _observed_information(self, t, e):
        # Analytic Hessian on (log k, log lam), so the SEs are well behaved.
        k, lam = self.shape, self.scale
        u = np.log(t / lam)
        z = np.exp(k * u)
        r = e.sum()
        # first derivatives w.r.t. a = log k, b = log lam
        # l = r a - r k b + (k - 1) sum_e log t - sum z ; dz/da = k u z ; dz/db = -k z
        h_aa = float(np.sum(e * k * u) - np.sum(k * u * z * (1 + k * u)))
        h_bb = float(-np.sum(k * k * z))
        h_ab = float(-r * k + np.sum(k * z * (1 + k * u)))
        H = np.array([[h_aa, h_ab], [h_ab, h_bb]])
        cov = np.linalg.inv(-H)
        self.log_shape_se = float(math.sqrt(cov[0, 0]))
        self.log_scale_se = float(math.sqrt(cov[1, 1]))
        zq = norm_ppf(1 - self.alpha / 2)
        self.shape_ci = (k * math.exp(-zq * self.log_shape_se), k * math.exp(zq * self.log_shape_se))
        self.scale_ci = (lam * math.exp(-zq * self.log_scale_se), lam * math.exp(zq * self.log_scale_se))

    def survival_at(self, t):
        return np.exp(-(np.asarray(t, float) / self.scale) ** self.shape)

    def hazard_at(self, t):
        t = np.asarray(t, float)
        return self.shape / self.scale * (t / self.scale) ** (self.shape - 1)

    @property
    def median(self) -> float:
        return self.scale * math.log(2) ** (1.0 / self.shape)
