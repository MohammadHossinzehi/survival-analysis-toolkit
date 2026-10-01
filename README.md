# survival-analysis-toolkit

`survkit` is survival analysis written from first principles in Python, with NumPy as the only dependency. It covers the workhorse methods of clinical trials, reliability engineering and churn modelling: Kaplan Meier curves with proper confidence bands, the weighted log rank family of tests, Cox proportional hazards regression (Efron and Breslow ties, strata, ridge penalty, residuals and a proportional hazards check), parametric Weibull fits, and the two evaluation metrics people actually report, Harrell's C index and the IPCW Brier score.

Every headline number is checked against published reference values for the classic Gehan leukaemia trial and, when it is installed, cross checked against [lifelines](https://github.com/CamDavidsonPilon/lifelines) to 1e-5 or better.

## Why survival analysis needs its own tools

Time to event data is almost never complete. A patient leaves the study, a machine is still running when you pull the logs, a customer has not churned *yet*. Those observations are **censored**: you know the event had not happened by time t, not when it happens. Dropping them biases everything toward short times; treating them as events is worse. The methods here use the partial information correctly, which is the whole point of the field.

## Quick look

```
$ python examples/gehan_walkthrough.py
6MP: median 23 weeks (95% CI 13 to inf), RMST(23) = 17.91 +/- 1.55
      time  at_risk  events  survival   std_err     lower     upper
         6       21       3    0.8571    0.0764    0.6197    0.9516
         7       17       1    0.8067    0.0869    0.5631    0.9228
        ...
1.00 |xxooooooooo
     |  xx       o
     |    xx      oooooo
0.75 |      x           ooooo
     |         xxxxx                        oo
0.50 |                                        ooooooooooooooooooooo
     |              xxxxx
0.25 |                     xxxxx
     |                          xxxxxxxxxxxx
0.00 |                                        xxxxxxxxxxxxxxxxxxxxx
      o = 6MP    x = placebo

logrank test: chi2 = 16.7929 on 1 df, p = 4.169e-05

Cox PH  n=42  events=30  ties=efron  penalizer=0  iterations=4
covariate           coef exp(coef)        se        z         p    lo 95%    hi 95%
placebo           1.5721    4.8169    0.4124    3.812  0.000138    2.1465   10.8093
likelihood ratio  =   16.3517 on 1 df, p = 5.261e-05
```

Children on placebo relapse at almost five times the rate of those on 6MP, and the curve plot (drawn in plain text so the demo needs no plotting library) shows it.

## Using it

```python
import numpy as np
import survkit as sk

weeks, relapsed, placebo = sk.load_gehan()

km = sk.KaplanMeier().fit(weeks, relapsed)
km.survival_at([5, 10, 20])          # step function lookup
km.median, km.median_confidence_interval()
km.restricted_mean(tau=20)           # RMST and its standard error

sk.logrank_test(weeks, relapsed, groups=placebo)                     # classic
sk.logrank_test(weeks, relapsed, groups=placebo, weighting="peto")   # early differences
sk.logrank_test(weeks, relapsed, groups=placebo,
                weighting="fleming_harrington", p=0, q=1)            # late differences

cox = sk.CoxPH(ties="efron").fit(placebo, weeks, relapsed, feature_names=["placebo"])
print(cox.summary())
cox.predict_survival_function(np.array([[0], [1]]), times=[5, 10, 20])
cox.check_proportional_hazards("km")   # scaled Schoenfeld residual test per covariate

X, t, e, _ = sk.simulate_weibull_ph(2000, beta=[0.8, -0.5], rng=0)
model = sk.CoxPH(penalizer=0.1).fit(X, t, e)
sk.concordance_index(t, -model.predict_log_partial_hazard(X), e)  # larger score = longer survival
sk.CoxPH().fit(X[:, 1:], t, e, strata=(X[:, 0] > 0))              # separate baseline per stratum
sk.WeibullFitter().fit(t, e).shape
```

## What is in the box

| Module | Contents |
| --- | --- |
| `nonparametric` | Kaplan Meier (Greenwood variance, log(-log S) bands, median and its CI, RMST with SE), Nelson Aalen, event tables. Delayed entry (left truncation) supported. |
| `tests_stats` | K sample weighted log rank: `logrank`, `wilcoxon` (Gehan and Breslow), `tarone_ware`, `peto`, `fleming_harrington(p, q)`. Log rank test for trend over ordered groups. |
| `cox` | Cox PH with Efron or Breslow ties, strata, ridge penalty, LR / Wald / score tests, Breslow baseline hazard, survival and median prediction, Schoenfeld and martingale residuals, Grambsch and Therneau style PH test with rank, identity, log and KM time transforms. |
| `parametric` | Weibull (profile likelihood, analytic observed information) and exponential fits. |
| `metrics` | Harrell's C index in O(n log n), IPCW Brier score and integrated Brier score. |
| `datasets` | The Gehan 6MP trial and a Weibull proportional hazards simulator. |
| `_stats` | Chi square survival function (incomplete gamma via series and Lentz continued fraction), normal CDF and quantile, so SciPy is not needed. |

## Running it

Python 3.9+ and NumPy.

```
pip install numpy pytest
python examples/gehan_walkthrough.py
python -m pytest
```

Or install the package with `pip install .` and `import survkit` from anywhere. To also run the lifelines cross check suite, `pip install lifelines pandas`; without them those tests are skipped automatically.

## Design notes

**Cox likelihood in one pass.** The usual textbook loop recomputes every risk set from scratch, which is O(n²). Here the subjects are sorted by time once per stratum, and the risk set sums of `exp(x'b)`, `exp(x'b) x` and `exp(x'b) x x'` come from reverse cumulative sums, read off at the first index of each distinct event time. Tied deaths are aggregated with `np.add.at`, and Efron's correction is vectorised by expanding each event time into `d` rows with fractions `0, 1/d, ..., (d-1)/d`. Setting the fractions to zero gives Breslow, so both approximations share the same code. A 50,000 row, 5 covariate model fits in under a second.

**Numerical safety.** The linear predictor is shifted by its maximum before exponentiating, covariates are centred (the coefficients are unchanged, and a test checks that), and Newton steps are halved until the partial likelihood does not decrease. Under complete separation the MLE is at infinity; the fit warns instead of silently returning a huge number, and the ridge penalty is there to fix it.

**Weibull without a 2D optimiser.** For fixed shape k the scale has a closed form, which leaves a one dimensional score equation that is strictly increasing in k. It is bracketed, then solved by Newton steps that fall back to bisection whenever they leave the bracket, so it always converges. Standard errors come from the analytic Hessian on the (log k, log scale) parameterisation.

**C index in O(n log n).** Subjects are swept from the longest time to the shortest while a Fenwick tree over prediction ranks counts how many later subjects scored higher, equal or lower. A subject censored at the exact time of a death counts as having outlived it, which matches Harrell's rule and lifelines.

**Choices worth knowing about.**

* Confidence bands use the log(-log S) transform (R's `conf.type = "log-log"`), not R's default plain log, because they cannot leave [0, 1].
* The baseline hazard is the Breslow estimator for both tie methods and predictions are true step functions (lifelines interpolates between event times, so its predicted curves differ slightly between events).
* The ridge penalty is `0.5 * penalizer * ||b||²` on the raw covariate scale and the summed (not averaged) log likelihood.

## Testing

70 tests, in three layers:

* **Reference values.** The Gehan 6MP Kaplan Meier table (survival, Greenwood SE and CI at every event time), group medians, the log rank chi square of 16.79, and the Cox coefficient, SE and all three global tests for both Efron (1.5721) and Breslow (1.5092) ties, as reported by R's `survival` package.
* **Mathematical identities and properties.** Analytic gradient and Hessian match finite differences on heavily tied data; the log rank statistic equals the Cox score test when there are no ties; martingale residuals sum to zero and Schoenfeld residuals sum to zero at the MLE; KM without censoring equals the empirical survival function; KM with delayed entry matches a brute force risk set count; the fast C index equals the O(n²) definition; the IPCW Brier score recovers the uncensored Brier score; null log rank p values are close to uniform; true Cox and Weibull parameters are recovered from simulated data; a time varying effect is flagged by the PH test while a proportional one is not.
* **Cross check against lifelines** (optional): KM, Nelson Aalen, all five log rank weightings, Cox coefficients, SEs, log likelihood and concordance (plain and stratified), all four PH test transforms, and the Weibull MLE.

## Data

The Gehan dataset is the 6 mercaptopurine leukaemia remission trial (Freireich et al., *Blood* 1963; Gehan, *Biometrika* 1965), reproduced in nearly every survival analysis text.

## License

MIT
