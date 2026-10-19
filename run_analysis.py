"""End-to-end geo lift analysis on the synthetic panel.

Run:  python run_analysis.py
Writes charts to ./charts and prints a short readout.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.simulate import simulate, TEST_MARKETS, TRUE_LIFT
from src import geolift as gl

Path("data").mkdir(exist_ok=True)
Path("charts").mkdir(exist_ok=True)

df = simulate()
df.to_csv("data/geo_panel.csv", index=False)

wide = gl.to_wide(df)
period = gl._periods(df)
test = TEST_MARKETS
control = [m for m in wide.columns if m not in test]

did = gl.scaled_control(wide, period, test, control)
sc, weights = gl.synthetic_control(wide, period, test, control)
did_post = gl.scaled_control(wide, period, test, control, window="post")
null = gl.placebo_in_space(wide, period, control, n_test=len(test))
p_value = float((null >= did.lift_pct).mean())
power = gl.power_curve(wide, period, control, n_test=len(test))

# ---------- readout ----------
lines = [
    "GEO LIFT READOUT (synthetic data)",
    f"Test markets: {', '.join(test)}  |  Control markets: {len(control)}",
    f"Simulated true lift during flight: ~{TRUE_LIFT:.0%} (after a 2-week ramp)",
    "",
    f"{'Method':<24}{'Lift':>8}{'Incremental visits':>22}{'Pre-period MAPE':>18}",
]
for r in (did, sc):
    lines.append(f"{r.method:<24}{r.lift_pct:>8.1%}{r.incremental:>22,.0f}{r.pre_fit_mape:>18.1%}")
lines += [
    "",
    f"Placebo test: {len(null)} fake tests on control markets; "
    f"95th percentile of placebo lift = {np.quantile(null, 0.95):.1%}; p = {p_value:.3f}",
    f"Carryover (6 weeks after flight): {did_post.lift_pct:.1%} lift, {did_post.incremental:,.0f} extra visits",
    "",
    "Synthetic control weights: " + ", ".join(f"{m} {w:.0%}" for m, w in weights.head(6).items()),
    "",
    "Power with this design (5 test markets, 8-week flight):",
]
lines += [f"  true lift {r.true_lift:>4.0%}  ->  detected {r.power:>5.0%} of the time" for r in power.itertuples()]
readout = "\n".join(lines)
print(readout)
Path("charts/readout.txt").write_text(readout)

# ---------- charts ----------
INK, MUTED, ACCENT, SHADE = "#1f2a37", "#8a94a3", "#c2410c", "#f1e7df"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})

def shade_flight(ax, s):
    weeks = s.index
    tw = weeks[(s["period"] == "test").values]
    ax.axvspan(tw[0], tw[-1], color=SHADE, zorder=0, label="Campaign flight")

# 1. Actual vs counterfactual (synthetic control)
s = sc.series.copy()
s.index = pd.to_datetime(s.index)
fig, ax = plt.subplots(figsize=(9, 4))
shade_flight(ax, s)
ax.plot(s.index, s["counterfactual"], color=MUTED, lw=2, ls="--", label="Counterfactual (synthetic control)")
ax.plot(s.index, s["actual"], color=INK, lw=2, label="Test markets, actual")
ax.set_ylabel("Weekly site visits")
ax.set_title("Test markets vs. what we'd expect without the campaign", loc="left", color=INK)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig("charts/actual_vs_counterfactual.png", dpi=150); plt.close(fig)

# 2. Cumulative incremental visits
gap = (s["actual"] - s["counterfactual"]).where(s["period"] != "pre", 0).cumsum()
fig, ax = plt.subplots(figsize=(9, 3.4))
shade_flight(ax, s)
ax.plot(s.index, gap, color=ACCENT, lw=2.2)
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_ylabel("Cumulative extra visits")
ax.set_title("Incremental visits build during the flight and keep accruing after it (carryover)", loc="left", color=INK)
fig.tight_layout(); fig.savefig("charts/cumulative_incremental.png", dpi=150); plt.close(fig)

# 3. Placebo distribution
fig, ax = plt.subplots(figsize=(9, 3.4))
ax.hist(null * 100, bins=30, color=MUTED, alpha=0.8, label="Placebo lifts (fake tests)")
ax.axvline(did.lift_pct * 100, color=ACCENT, lw=2.5, label=f"Observed lift {did.lift_pct:.1%}")
ax.set_xlabel("Estimated lift (%)"); ax.set_ylabel("Count")
ax.set_title("The observed lift sits far outside what noise alone produces", loc="left", color=INK)
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("charts/placebo_distribution.png", dpi=150); plt.close(fig)

# 4. Power curve
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.plot(power.true_lift * 100, power.power * 100, color=INK, marker="o")
ax.axhline(80, color=ACCENT, ls="--", lw=1.2, label="80% power")
ax.set_xlabel("True lift (%)"); ax.set_ylabel("Chance of detecting it (%)")
ax.set_ylim(0, 105)
ax.set_title("Minimum detectable lift for this design", loc="left", color=INK)
ax.legend(frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig("charts/power_curve.png", dpi=150); plt.close(fig)

print("\ncharts written to ./charts")
