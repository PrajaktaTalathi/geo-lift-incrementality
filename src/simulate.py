"""Generate a synthetic weekly panel of site visits by market for a geo test.

The shape mirrors a real offline (radio / out-of-home) brand test: a set of
markets with different sizes, a shared seasonal cycle, market-level noise, and
a campaign that runs in a few test markets for a fixed number of weeks, with a
fading carryover effect afterwards.

All numbers here are synthetic. No real company data is used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRE_WEEKS = 52
TEST_WEEKS = 8
POST_WEEKS = 6
N_MARKETS = 20
TEST_MARKETS = ["M03", "M07", "M11", "M14", "M18"]
TRUE_LIFT = 0.12       # 12% lift on visits during the flight
CARRYOVER_DECAY = 0.55  # share of the lift that survives each week after the flight


def simulate(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    weeks = PRE_WEEKS + TEST_WEEKS + POST_WEEKS
    week_idx = np.arange(weeks)
    start = pd.Timestamp("2025-01-06")  # a Monday
    dates = start + pd.to_timedelta(week_idx * 7, unit="D")

    # Shared seasonality: two demand peaks a year (think an enrolment cycle)
    season = 1 + 0.25 * np.sin(2 * np.pi * (week_idx - 20) / 52) + 0.10 * np.sin(4 * np.pi * week_idx / 52)
    trend = 1 + 0.0015 * week_idx

    markets = [f"M{i:02d}" for i in range(1, N_MARKETS + 1)]
    base_size = rng.lognormal(mean=8.3, sigma=0.45, size=N_MARKETS)  # weekly visits ~ 2k-10k
    market_seasonality = rng.normal(1.0, 0.08, size=N_MARKETS)       # each market follows the cycle a bit differently

    flight_start = PRE_WEEKS
    flight_end = PRE_WEEKS + TEST_WEEKS
    lift_curve = np.zeros(weeks)
    lift_curve[flight_start:flight_end] = TRUE_LIFT * np.minimum(1, (np.arange(TEST_WEEKS) + 1) / 2)  # ramps up over 2 weeks
    for k in range(POST_WEEKS):
        lift_curve[flight_end + k] = TRUE_LIFT * CARRYOVER_DECAY ** (k + 1)

    rows = []
    for m, size, ms in zip(markets, base_size, market_seasonality):
        expected = size * (1 + ms * (season - 1)) * trend
        noise = rng.normal(0, 0.08, size=weeks)
        treated = m in TEST_MARKETS
        visits = expected * (1 + noise) * (1 + (lift_curve if treated else 0))
        for w in range(weeks):
            period = "pre" if w < flight_start else ("test" if w < flight_end else "post")
            rows.append({
                "week": dates[w].date().isoformat(),
                "market": m,
                "group": "test" if treated else "control",
                "period": period,
                "visits": int(round(visits[w])),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = simulate()
    df.to_csv("data/geo_panel.csv", index=False)
    print(f"wrote data/geo_panel.csv: {len(df)} rows, {df.market.nunique()} markets")
