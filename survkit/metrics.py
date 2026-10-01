"""Evaluation metrics for survival predictions under right censoring."""
from __future__ import annotations

import numpy as np

from ._utils import check_survival_data

__all__ = ["concordance_index", "brier_score", "integrated_brier_score"]


def concordance_index(durations, predicted_survival_scores, events=None) -> float:
    """Harrell's C index.

    ``predicted_survival_scores`` should be larger for subjects expected to
    live longer (pass a negative risk score for hazard type predictions). A
    pair (i, j) is comparable when i has an observed event and either
    t_i < t_j, or t_i == t_j with j censored (a subject censored at the time of
    a death is taken to have outlived it). Two deaths at the same time are not
    comparable. Ties in the prediction count one half.

    Runs in O(n log n): subjects are swept in time order while a Fenwick tree
    over prediction ranks counts how many later time subjects have a larger,
    equal or smaller score.
    """
    t, e, _ = check_survival_data(durations, events)
    s = np.asarray(predicted_survival_scores, dtype=float).ravel()
    if s.shape != t.shape:
        raise ValueError("predictions must have the same length as durations")
    uniq, rank = np.unique(s, return_inverse=True)
    m = uniq.size
    tree = np.zeros(m + 1)

    def add(i):
        i += 1
        while i <= m:
            tree[i] += 1
            i += i & -i

    def prefix(i):  # number of inserted ranks < i
        total = 0.0
        while i > 0:
            total += tree[i]
            i -= i & -i
        return total

    order = np.argsort(-t, kind="mergesort")  # longest time first
    concordant = 0.0
    comparable = 0.0
    inserted = 0
    k = 0
    n = t.size
    while k < n:
        # process one block of tied times at a time
        j = k
        while j + 1 < n and t[order[j + 1]] == t[order[k]]:
            j += 1
        block = order[k:j + 1]
        # Censored subjects tied with a death at this time outlived it, so they
        # join the comparison set before the deaths are scored (Harrell's rule).
        cens = [i for i in block if not e[i]]
        for i in cens:
            add(rank[i])
        inserted += len(cens)
        for i in block:
            if not e[i] or inserted == 0:
                continue
            r = rank[i]
            below = prefix(r)
            equal = prefix(r + 1) - below
            above = inserted - below - equal
            # subject i died first, so it should have the lower survival score
            concordant += above + 0.5 * equal
            comparable += inserted
        deaths = [i for i in block if e[i]]
        for i in deaths:
            add(rank[i])
        inserted += len(deaths)
        k = j + 1
    if comparable == 0:
        raise ValueError("no comparable pairs")
    return concordant / comparable


def _censoring_km(t, e):
    """Kaplan Meier of the censoring distribution G(u) = P(C > u)."""
    from .nonparametric import KaplanMeier
    return KaplanMeier().fit(t, ~e)


def brier_score(durations, events, survival_at_t, t: float) -> float:
    """Inverse probability of censoring weighted Brier score at time ``t``.

    ``survival_at_t[i]`` is the predicted P(T_i > t). Following Graf et al.
    (1999), subjects who died before t are weighted by 1 / G(T_i-), subjects
    still alive at t by 1 / G(t), and subjects censored before t contribute
    nothing (their weight is redistributed by G).
    """
    tt, e, _ = check_survival_data(durations, events)
    p = np.asarray(survival_at_t, dtype=float).ravel()
    if p.shape != tt.shape:
        raise ValueError("survival_at_t must have one prediction per subject")
    G = _censoring_km(tt, e)
    died = (tt <= t) & e
    alive = tt > t
    g_died = np.asarray(G.survival_at(tt, left_limit=True), float)
    g_t = float(G.survival_at(t))
    with np.errstate(divide="ignore", invalid="ignore"):
        term_died = np.where(died, p ** 2 / g_died, 0.0)
        term_alive = np.where(alive, (1 - p) ** 2 / g_t, 0.0) if g_t > 0 else np.zeros_like(p)
    return float(np.mean(term_died + term_alive))


def integrated_brier_score(durations, events, survival_matrix, times) -> float:
    """Trapezoidal average of the Brier score over ``times``.

    ``survival_matrix`` has shape (len(times), n_subjects), the layout that
    ``CoxPH.predict_survival_function`` returns.
    """
    times = np.asarray(times, dtype=float)
    S = np.asarray(survival_matrix, dtype=float)
    if S.shape[0] != times.size:
        raise ValueError("survival_matrix rows must match times")
    if times.size < 2:
        raise ValueError("need at least two time points")
    scores = np.array([brier_score(durations, events, S[k], tk) for k, tk in enumerate(times)])
    area = np.sum(0.5 * (scores[1:] + scores[:-1]) * np.diff(times))
    return float(area / (times[-1] - times[0]))
