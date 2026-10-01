"""Cox proportional hazards regression.

The partial likelihood, its gradient and its Hessian are computed in a single
O(n p^2) pass per Newton step: subjects are sorted by time once, risk set sums
come from reverse cumulative sums, and tied event times are handled with
either Breslow's or Efron's approximation. Stratification multiplies the
partial likelihoods of the strata (each stratum gets its own baseline
hazard, the coefficients are shared).
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np

from ._stats import chi2_sf, norm_ppf, norm_sf
from ._utils import check_survival_data
from .metrics import concordance_index
from .nonparametric import KaplanMeier

__all__ = ["CoxPH", "ConvergenceWarning"]


class ConvergenceWarning(UserWarning):
    pass


@dataclass
class _Stratum:
    X: np.ndarray            # covariates sorted by time (centred)
    start: np.ndarray        # index of first subject at risk for each event time
    event_time: np.ndarray   # distinct event times
    rep: np.ndarray          # event time index, repeated once per tied event
    frac: np.ndarray         # Efron fraction l / d for each repeated row
    event_X_sum: np.ndarray  # sum of covariates of events, shape (p,)
    tie_index: np.ndarray    # for each event row (sorted), its event time index
    event_rows: np.ndarray   # positions (in sorted order) of the events
    deaths: np.ndarray       # number of events at each distinct event time


def _rev_cumsum(a):
    return np.flip(np.cumsum(np.flip(a, axis=0), axis=0), axis=0)


class CoxPH:
    """Semiparametric proportional hazards model h(t|x) = h0(t) exp(x'b).

    Parameters
    ----------
    ties : {"efron", "breslow"}
        Approximation for tied event times. Efron is the more accurate one and
        the default in R's ``survival`` package.
    penalizer : float
        Ridge penalty 0.5 * penalizer * ||b||^2 subtracted from the log partial
        likelihood. Useful under (quasi) separation or with many covariates.
    alpha : float
        1 - confidence level for coefficient intervals.
    """

    def __init__(self, ties: str = "efron", penalizer: float = 0.0, alpha: float = 0.05,
                 tol: float = 1e-10, max_iter: int = 100):
        if ties not in ("efron", "breslow"):
            raise ValueError("ties must be 'efron' or 'breslow'")
        if penalizer < 0:
            raise ValueError("penalizer must be nonnegative")
        self.ties = ties
        self.penalizer = float(penalizer)
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter

    # ------------------------------------------------------------------ setup
    def _prepare(self, Xc, t, e, strata):
        out = []
        for lab in np.unique(strata):
            m = np.nonzero(strata == lab)[0]
            order = m[np.argsort(t[m], kind="mergesort")]
            ts, es, Xs = t[order], e[order], Xc[order]
            event_time, deaths = np.unique(ts[es], return_counts=True)
            if event_time.size == 0:
                continue
            start = np.searchsorted(ts, event_time, side="left")
            event_rows = np.nonzero(es)[0]
            tie_index = np.searchsorted(event_time, ts[event_rows])
            rep = np.repeat(np.arange(event_time.size), deaths)
            if self.ties == "efron":
                frac = np.concatenate([np.arange(d) / d for d in deaths])
            else:
                frac = np.zeros(rep.size)
            out.append(_Stratum(Xs, start, event_time, rep, frac, Xs[event_rows].sum(axis=0),
                                tie_index, event_rows, deaths))
        if not out:
            raise ValueError("no events observed; the Cox model is not identifiable")
        return out

    # ------------------------------------------------------- core likelihood
    @staticmethod
    def _stratum_terms(beta, s: _Stratum, need_hess=True):
        X = s.X
        eta = X @ beta
        c = eta.max()
        w = np.exp(eta - c)
        r0 = _rev_cumsum(w)[s.start]
        r1 = _rev_cumsum(w[:, None] * X)[s.start]
        k = s.event_time.size
        p = X.shape[1]
        ev_w = w[s.event_rows]
        d0 = np.bincount(s.tie_index, weights=ev_w, minlength=k)
        d1 = np.zeros((k, p))
        np.add.at(d1, s.tie_index, ev_w[:, None] * X[s.event_rows])

        f = s.frac
        den = r0[s.rep] - f * d0[s.rep]
        num1 = r1[s.rep] - f[:, None] * d1[s.rep]
        ll = float(eta[s.event_rows].sum() - np.sum(np.log(den) + c))
        grad = s.event_X_sum - (num1 / den[:, None]).sum(axis=0)
        if not need_hess:
            return ll, grad, None
        outer = X[:, :, None] * X[:, None, :]
        r2 = _rev_cumsum(w[:, None, None] * outer)[s.start]
        d2 = np.zeros((k, p, p))
        np.add.at(d2, s.tie_index, ev_w[:, None, None] * outer[s.event_rows])
        num2 = r2[s.rep] - f[:, None, None] * d2[s.rep]
        mean1 = num1 / den[:, None]
        hess = -(np.sum(num2 / den[:, None, None], axis=0)
                 - np.einsum("ri,rj->ij", mean1, mean1))
        return ll, grad, hess

    def _loglik(self, beta, strata, penalized=True, need_hess=True):
        p = beta.size
        ll, g, H = 0.0, np.zeros(p), np.zeros((p, p))
        for s in strata:
            a, b, c = self._stratum_terms(beta, s, need_hess)
            ll += a
            g += b
            if need_hess:
                H += c
        if penalized and self.penalizer > 0:
            ll -= 0.5 * self.penalizer * beta @ beta
            g -= self.penalizer * beta
            H -= self.penalizer * np.eye(p)
        return ll, g, H

    # -------------------------------------------------------------- fitting
    def fit(self, X, durations, events=None, strata=None, feature_names=None) -> "CoxPH":
        t, e, _ = check_survival_data(durations, events)
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        if X.shape[0] != t.size:
            raise ValueError("X and durations have different numbers of rows")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains non finite values")
        n, p = X.shape
        strata_arr = np.zeros(n, dtype=int) if strata is None else np.asarray(strata).ravel()
        if strata_arr.shape != (n,):
            raise ValueError("strata must have one label per row")

        self.feature_names = list(feature_names) if feature_names is not None else \
            [f"x{i}" for i in range(p)]
        if len(self.feature_names) != p:
            raise ValueError("feature_names has the wrong length")
        self._mean = X.mean(axis=0)
        Xc = X - self._mean
        prepared = self._prepare(Xc, t, e, strata_arr)

        beta = np.zeros(p)
        ll, g, H = self._loglik(beta, prepared)
        self.null_log_likelihood = self._loglik(beta, prepared, penalized=False, need_hess=False)[0]
        score0, info0 = g.copy(), -H.copy()
        converged = False
        for it in range(1, self.max_iter + 1):
            try:
                step = np.linalg.solve(-H, g)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(-H, g, rcond=None)[0]
            # Step halving keeps every iterate an ascent step.
            for _ in range(40):
                cand = beta + step
                ll_new, g_new, H_new = self._loglik(cand, prepared)
                if np.isfinite(ll_new) and ll_new >= ll - 1e-12 * (1 + abs(ll)):
                    break
                step = step / 2
            delta = ll_new - ll
            beta, ll, g, H = cand, ll_new, g_new, H_new
            if abs(delta) < self.tol * (1 + abs(ll)) and np.max(np.abs(step)) < 1e-6:
                converged = True
                break
        self.n_iter = it
        self.converged = converged
        if not converged:
            warnings.warn("Newton Raphson did not converge; coefficients may be diverging "
                          "(separation?). Consider a penalizer.", ConvergenceWarning)
        if np.max(np.abs(beta)) > 15:
            warnings.warn("very large coefficient: possible monotone likelihood / separation",
                          ConvergenceWarning)

        info = -H
        try:
            cov = np.linalg.inv(info)
        except np.linalg.LinAlgError:
            # Happens under separation, where the likelihood is flat at infinity.
            cov = np.linalg.pinv(info)
        self.coef = beta
        self.variance_matrix = cov
        self.se = np.sqrt(np.diag(cov))
        self.log_likelihood = self._loglik(beta, prepared, penalized=False, need_hess=False)[0]
        self.penalized_log_likelihood = ll
        z = norm_ppf(1 - self.alpha / 2)
        self.z = beta / self.se
        self.p_values = np.array([2 * norm_sf(abs(v)) for v in self.z])
        self.hazard_ratios = np.exp(beta)
        self.hr_lower = np.exp(beta - z * self.se)
        self.hr_upper = np.exp(beta + z * self.se)

        lr = 2 * (self.log_likelihood - self.null_log_likelihood)
        wald = float(beta @ info @ beta)
        score = float(score0 @ np.linalg.lstsq(info0, score0, rcond=None)[0])
        self.tests = {
            "likelihood_ratio": (lr, p, chi2_sf(lr, p)),
            "wald": (wald, p, chi2_sf(wald, p)),
            "score": (score, p, chi2_sf(score, p)),
        }

        self._strata_labels = np.unique(strata_arr)
        self._stratified = strata is not None
        self._fit_baseline(prepared, strata_arr, t, e)
        self._train = (X, t, e, strata_arr)
        self._prepared = prepared
        self.n_obs, self.n_events = n, int(e.sum())
        self.concordance = concordance_index(t, -self.predict_log_partial_hazard(X), e)
        return self

    def _fit_baseline(self, prepared, strata_arr, t, e):
        """Breslow estimator of the baseline cumulative hazard per stratum."""
        self.baseline = {}
        labels = [lab for lab in np.unique(strata_arr) if np.any(e[strata_arr == lab])]
        for lab, s in zip(labels, prepared):
            w = np.exp(s.X @ self.coef)
            r0 = _rev_cumsum(w)[s.start]
            H0 = np.cumsum(s.deaths / r0)
            self.baseline[lab] = (s.event_time, H0)

    # ----------------------------------------------------------- prediction
    def _check(self):
        if not hasattr(self, "coef"):
            raise RuntimeError("call fit() first")

    def _as_matrix(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :] if X.size == self.coef.size and self.coef.size > 1 else X[:, None]
        return X

    def predict_log_partial_hazard(self, X):
        self._check()
        return (self._as_matrix(X) - self._mean) @ self.coef

    def predict_partial_hazard(self, X):
        return np.exp(self.predict_log_partial_hazard(X))

    def baseline_cumulative_hazard(self, times, stratum=None):
        self._check()
        if stratum is None:
            if self._stratified:
                raise ValueError("model is stratified: pass stratum")
            stratum = self._strata_labels[0]
        if stratum not in self.baseline:
            raise KeyError(f"unknown stratum {stratum!r}")
        grid, H0 = self.baseline[stratum]
        idx = np.searchsorted(grid, np.asarray(times, float), side="right") - 1
        return np.where(idx >= 0, H0[np.clip(idx, 0, None)], 0.0)

    def predict_cumulative_hazard(self, X, times, strata=None):
        """Matrix of H(t | x) with shape (len(times), n_subjects)."""
        X = self._as_matrix(X)
        times = np.atleast_1d(np.asarray(times, float))
        ph = self.predict_partial_hazard(X)
        if strata is None:
            base = self.baseline_cumulative_hazard(times)[:, None]
            return base * ph[None, :]
        strata = np.asarray(strata).ravel()
        out = np.empty((times.size, X.shape[0]))
        for i, lab in enumerate(strata):
            out[:, i] = self.baseline_cumulative_hazard(times, lab) * ph[i]
        return out

    def predict_survival_function(self, X, times, strata=None):
        return np.exp(-self.predict_cumulative_hazard(X, times, strata))

    def predict_median(self, X, strata=None):
        """Median survival time per subject (inf when S never reaches 0.5)."""
        X = self._as_matrix(X)
        out = np.empty(X.shape[0])
        labs = [None] * X.shape[0] if strata is None else list(np.asarray(strata).ravel())
        for i in range(X.shape[0]):
            key = labs[i] if labs[i] is not None else self._strata_labels[0]
            grid, _ = self.baseline[key]
            surv = self.predict_survival_function(X[i:i + 1], grid,
                                                  None if labs[i] is None else [labs[i]])[:, 0]
            hit = np.nonzero(surv <= 0.5)[0]
            out[i] = grid[hit[0]] if hit.size else np.inf
        return out

    # ------------------------------------------------------------ residuals
    def schoenfeld_residuals(self):
        """Unscaled Schoenfeld residuals, one row per event.

        Returns (event_times, residuals) where residual = x_i - E[x | risk set]
        under the fitted model. Events are pooled across strata.
        """
        self._check()
        times, res = [], []
        for s in self._prepared:
            w = np.exp(s.X @ self.coef)
            r0 = _rev_cumsum(w)[s.start]
            r1 = _rev_cumsum(w[:, None] * s.X)[s.start]
            mean = r1 / r0[:, None]
            res.append(s.X[s.event_rows] - mean[s.tie_index])
            times.append(s.event_time[s.tie_index])
        times = np.concatenate(times)
        res = np.vstack(res)
        order = np.argsort(times, kind="mergesort")
        return times[order], res[order]

    def martingale_residuals(self):
        """delta_i - H0(t_i) exp(x_i'b): observed minus model expected events."""
        self._check()
        X, t, e, strata = self._train
        if self._stratified:
            H = np.array([self.baseline_cumulative_hazard([ti], si)[0] for ti, si in zip(t, strata)])
        else:
            H = self.baseline_cumulative_hazard(t)
        return e.astype(float) - H * self.predict_partial_hazard(X)

    def check_proportional_hazards(self, transform: str = "rank"):
        """Grambsch and Therneau style test of the proportional hazards assumption.

        Scaled Schoenfeld residuals are correlated with a transform g(t) of
        event time. Under PH they have no trend, so each covariate's statistic
        is chi square with one degree of freedom. Returns a dict mapping feature
        name to (statistic, p_value).
        """
        times, r = self.schoenfeld_residuals()
        n_events = r.shape[0]
        if transform == "rank":
            from ._ranks import average_ranks
            g = average_ranks(times)
        elif transform == "identity":
            g = times.astype(float)
        elif transform == "log":
            g = np.log(times)
        elif transform == "km":
            _, t, e, _ = self._train
            km = KaplanMeier().fit(t, e)
            g = 1.0 - km.survival_at(times)
        else:
            raise ValueError("transform must be 'rank', 'identity', 'log' or 'km'")
        g = g - g.mean()
        scaled = n_events * r @ self.variance_matrix
        stats = (g @ scaled) ** 2 / (n_events * np.diag(self.variance_matrix) * (g @ g))
        return {name: (float(s), chi2_sf(float(s), 1)) for name, s in zip(self.feature_names, stats)}

    # -------------------------------------------------------------- display
    def summary(self) -> str:
        self._check()
        lvl = int(round(100 * (1 - self.alpha)))
        head = (f"Cox PH  n={self.n_obs}  events={self.n_events}  ties={self.ties}"
                f"  penalizer={self.penalizer:g}  iterations={self.n_iter}")
        lines = [head, "",
                 "{:<14} {:>9} {:>9} {:>9} {:>8} {:>9} {:>9} {:>9}".format(
                     "covariate", "coef", "exp(coef)", "se", "z", "p", f"lo {lvl}%", f"hi {lvl}%")]
        for i, name in enumerate(self.feature_names):
            lines.append("{:<14} {:>9.4f} {:>9.4f} {:>9.4f} {:>8.3f} {:>9.3g} {:>9.4f} {:>9.4f}".format(
                name[:14], self.coef[i], self.hazard_ratios[i], self.se[i], self.z[i],
                self.p_values[i], self.hr_lower[i], self.hr_upper[i]))
        lines.append("")
        lines.append(f"log partial likelihood = {self.log_likelihood:.4f}   "
                     f"concordance = {self.concordance:.4f}")
        for k, (stat, df, pv) in self.tests.items():
            lines.append(f"{k.replace('_', ' '):<17} = {stat:9.4f} on {df} df, p = {pv:.4g}")
        return "\n".join(lines)
