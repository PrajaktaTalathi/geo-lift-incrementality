"""Geo test/control lift estimation.

Two estimators, from simplest to more robust:

1. Scaled control (difference-in-differences on a ratio scale): the
   counterfactual for the test markets is the control total during the flight,
   scaled by the test/control ratio seen in the pre-period.
2. Synthetic control: a non-negative weighted blend of control markets that
   best tracks the test markets in the pre-period, projected forward.

Inference uses placebo tests: re-run the same estimator pretending each
control market was the treated one. If the real lift sits far outside that
placebo distribution, it is unlikely to be noise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import nnls


@dataclass
class LiftResult:
    method: str
    actual: float           # test-market visits during the flight
    counterfactual: float   # what we estimate would have happened without the campaign
    incremental: float      # actual - counterfactual
    lift_pct: float         # incremental / counterfactual
    pre_fit_mape: float     # how well the counterfactual tracked the test markets before the flight
    series: pd.DataFrame    # weekly actual vs counterfactual


def to_wide(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot_table(index="week", columns="market", values="visits", aggfunc="sum").sort_index().astype(float)


def _periods(df: pd.DataFrame) -> pd.Series:
    return df.drop_duplicates("week").set_index("week")["period"].sort_index()


def _mape(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b) / a))


def scaled_control(wide: pd.DataFrame, period: pd.Series, test: list[str], control: list[str],
                   window: str = "test") -> LiftResult:
    t = wide[test].sum(axis=1)
    c = wide[control].sum(axis=1)
    pre = period == "pre"
    ratio = t[pre].sum() / c[pre].sum()
    cf = c * ratio
    w = period == window
    actual, counter = t[w].sum(), cf[w].sum()
    series = pd.DataFrame({"actual": t, "counterfactual": cf, "period": period})
    return LiftResult("Scaled control (DiD)", actual, counter, actual - counter,
                      (actual - counter) / counter, _mape(t[pre].values, cf[pre].values), series)


def synthetic_control(wide: pd.DataFrame, period: pd.Series, test: list[str], control: list[str],
                      window: str = "test") -> tuple[LiftResult, pd.Series]:
    t = wide[test].sum(axis=1)
    X = wide[control]
    pre = period == "pre"
    # Normalise columns so large markets don't dominate the fit, then solve non-negative least squares
    scale = X[pre].mean()
    weights, _ = nnls((X[pre] / scale).values, t[pre].values)
    cf = (X / scale).values @ weights
    cf = pd.Series(cf, index=wide.index)
    w = period == window
    actual, counter = t[w].sum(), cf[w].sum()
    series = pd.DataFrame({"actual": t, "counterfactual": cf, "period": period})
    wts = pd.Series(weights / weights.sum(), index=control).sort_values(ascending=False)
    res = LiftResult("Synthetic control", actual, counter, actual - counter,
                     (actual - counter) / counter, _mape(t[pre].values, cf[pre].values), series)
    return res, wts[wts > 0.001]


def placebo_in_space(wide: pd.DataFrame, period: pd.Series, control: list[str], n_test: int,
                     n_draws: int = 400, seed: int = 0) -> np.ndarray:
    """Lift estimates from fake tests: random sets of control markets treated as 'test'."""
    rng = np.random.default_rng(seed)
    lifts = []
    for _ in range(n_draws):
        fake = list(rng.choice(control, size=n_test, replace=False))
        rest = [m for m in control if m not in fake]
        lifts.append(scaled_control(wide, period, fake, rest).lift_pct)
    return np.array(lifts)


def power_curve(wide: pd.DataFrame, period: pd.Series, control: list[str], n_test: int,
                lifts=(0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10), n_draws: int = 300, seed: int = 1,
                alpha: float = 0.05) -> pd.DataFrame:
    """Before spending money: how often would a test of this design detect a lift of a given size?

    Injects a known lift into random 'fake test' sets of control markets during the
    flight window and checks how often it clears the placebo noise threshold.
    """
    rng = np.random.default_rng(seed)
    null = placebo_in_space(wide, period, control, n_test, n_draws=n_draws, seed=seed + 100)
    threshold = np.quantile(null, 1 - alpha)
    flight = (period == "test").values
    rows = []
    for lift in lifts:
        hits = 0
        for _ in range(n_draws):
            fake = list(rng.choice(control, size=n_test, replace=False))
            rest = [m for m in control if m not in fake]
            w2 = wide.astype(float)
            w2.loc[flight, fake] = w2.loc[flight, fake] * (1 + lift)
            if scaled_control(w2, period, fake, rest).lift_pct > threshold:
                hits += 1
        rows.append({"true_lift": lift, "power": hits / n_draws})
    return pd.DataFrame(rows)
