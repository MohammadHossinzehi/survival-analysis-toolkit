"""End to end analysis of the Gehan 6MP leukaemia trial.

Run from the repository root:  python examples/gehan_walkthrough.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from survkit import CoxPH, KaplanMeier, WeibullFitter, load_gehan, logrank_test  # noqa: E402


def ascii_km(curves, width=60, height=16, t_max=None):
    """Render step survival curves as text so the demo needs no plotting library."""
    t_max = t_max or max(km.timeline.max() for km, _ in curves)
    grid = np.zeros((height + 1, width + 1), dtype="<U1")
    grid[:] = " "
    for km, mark in curves:
        xs = np.linspace(0, t_max, width + 1)
        ys = np.atleast_1d(km.survival_at(xs))
        for col, y in enumerate(ys):
            grid[height - int(round(y * height)), col] = mark
    lines = []
    for r in range(height + 1):
        label = f"{1 - r / height:4.2f} |" if r % 4 == 0 else "     |"
        lines.append(label + "".join(grid[r]))
    lines.append("     +" + "-" * (width + 1))
    lines.append(f"      0{'weeks':^{width - 6}}{t_max:>5.0f}")
    return "\n".join(lines)


def main():
    weeks, relapsed, placebo = load_gehan()
    arms = {"6MP": placebo == 0, "placebo": placebo == 1}

    print("=" * 72)
    print("Kaplan Meier estimates")
    print("=" * 72)
    fits = {}
    for name, m in arms.items():
        km = KaplanMeier().fit(weeks[m], relapsed[m])
        fits[name] = km
        lo, hi = km.median_confidence_interval()
        rmst, se = km.restricted_mean(23)
        print(f"\n{name}: median {km.median:g} weeks (95% CI {lo:g} to {hi:g}),"
              f" RMST(23) = {rmst:.2f} +/- {se:.2f}")
        print(km.summary())

    print()
    print(ascii_km([(fits["6MP"], "o"), (fits["placebo"], "x")]))
    print("      o = 6MP    x = placebo")

    print("\n" + "=" * 72)
    print("Comparing the arms")
    print("=" * 72)
    for w in ("logrank", "wilcoxon", "peto"):
        print(logrank_test(weeks, relapsed, placebo, weighting=w), "\n")

    print("=" * 72)
    print("Cox proportional hazards")
    print("=" * 72)
    cox = CoxPH(ties="efron").fit(placebo, weeks, relapsed, feature_names=["placebo"])
    print(cox.summary())
    stat, p = cox.check_proportional_hazards("km")["placebo"]
    print(f"\nPH assumption (scaled Schoenfeld vs 1 - KM): chi2 = {stat:.3f}, p = {p:.3f}")

    print("\n" + "=" * 72)
    print("Parametric Weibull fit per arm")
    print("=" * 72)
    for name, m in arms.items():
        wf = WeibullFitter().fit(weeks[m], relapsed[m])
        lr, p = wf.exponential_lr_test
        print(f"{name:8s} shape {wf.shape:.3f} (95% CI {wf.shape_ci[0]:.2f}-{wf.shape_ci[1]:.2f}),"
              f" scale {wf.scale:.2f}, median {wf.median:.1f};"
              f" exponential vs Weibull LR p = {p:.3f}")


if __name__ == "__main__":
    main()
