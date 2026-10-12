"""Official targets T1-T6. Thresholds come from POOLED WARM-UP data only and are frozen.

Course alignment (Guidelines Ch. 7 and 12): thresholds are estimated once from the warm-up rows
of BOTH cities pooled, then frozen; values at t are used only to build the label y_t.
"""
import pandas as pd

RAIN_MM = 0.1
# target -> (variable, quantile, direction)
QUANTILE_TARGETS = {
    "T2": ("temperature", 0.80, ">="),
    "T3": ("temperature", 0.20, "<="),
    "T4": ("relative_humidity", 0.50, ">="),
    "T5": ("wind_speed", 0.80, ">="),
}


def compute_thresholds(df_a, df_b, warmup_n, target):
    """Estimate thresholds once from the first `warmup_n` rows of BOTH cities."""
    pooled = pd.concat([df_a.iloc[:warmup_n], df_b.iloc[:warmup_n]], ignore_index=True)
    thresholds = {}
    if target in QUANTILE_TARGETS:
        var, q, _ = QUANTILE_TARGETS[target]
        thresholds[target] = float(pooled[var].quantile(q))
    if target == "T6":  # rain OR strong wind -> needs the T5 wind threshold
        thresholds["T5"] = float(pooled["wind_speed"].quantile(0.80))
    return thresholds  # frozen from here on


def make_target(df, target, thresholds):
    """y_t computed from values observed at time t (used ONLY as the label)."""
    if target == "T1":
        y, src = df["precipitation"] >= RAIN_MM, ["precipitation"]
    elif target in QUANTILE_TARGETS:
        var, _, direction = QUANTILE_TARGETS[target]
        tau = thresholds[target]
        y = df[var] >= tau if direction == ">=" else df[var] <= tau
        src = [var]
    elif target == "T6":
        y = (df["precipitation"] >= RAIN_MM) | (df["wind_speed"] >= thresholds["T5"])
        src = ["precipitation", "wind_speed"]
    else:
        raise ValueError(f"Unknown target {target}")
    y = y.astype(float).where(df[src].notna().all(axis=1))  # keep missing as NaN
    return y.rename("y")
