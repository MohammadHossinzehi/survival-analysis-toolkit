from __future__ import annotations

import numpy as np


def check_survival_data(durations, events=None, entry=None):
    """Validate and coerce the (durations, events, entry) triple.

    ``events`` defaults to all ones (no censoring). ``entry`` is the delayed
    entry (left truncation) time and defaults to zero for every subject.
    """
    t = np.asarray(durations, dtype=float).ravel()
    if t.size == 0:
        raise ValueError("durations is empty")
    if not np.all(np.isfinite(t)):
        raise ValueError("durations must be finite")
    if np.any(t < 0):
        raise ValueError("durations must be nonnegative")
    if events is None:
        e = np.ones_like(t, dtype=bool)
    else:
        e_raw = np.asarray(events).ravel()
        if e_raw.shape != t.shape:
            raise ValueError("events must have the same length as durations")
        if not np.all(np.isin(e_raw, (0, 1, True, False))):
            raise ValueError("events must be 0/1 or boolean")
        e = e_raw.astype(bool)
    if entry is None:
        s = np.zeros_like(t)
    else:
        s = np.asarray(entry, dtype=float).ravel()
        if s.shape != t.shape:
            raise ValueError("entry must have the same length as durations")
        if np.any(s > t):
            raise ValueError("entry time must not exceed duration")
    return t, e, s


def at_risk_counts(times, durations, entry):
    """Number of subjects at risk just before each time in ``times``.

    A subject is at risk at ``u`` when ``entry < u <= duration``. Computed with
    two sorted searches, so it is O((n + m) log n).
    """
    d_sorted = np.sort(durations)
    s_sorted = np.sort(entry)
    still_in = d_sorted.size - np.searchsorted(d_sorted, times, side="left")
    entered_late = s_sorted.size - np.searchsorted(s_sorted, times, side="left")
    return still_in - entered_late
