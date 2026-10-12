"""Leakage-free feature representations. Everything ends at t-1.

Course alignment (Guidelines Ch. 8, W03):
- Representation A: previous state x(t-1).
- Representation B: A + past-only deltas, rolling mean/std, rain-hour counts and the rate of
  already-revealed past events. The 6/24-hour windows are our choice (config.yaml).
- Target-defining values at t (precipitation, wind speed) are never predictors at t.
"""
import pandas as pd

BASE = ["temperature", "relative_humidity", "precipitation", "wind_speed"]


def representation_a(df):
    """Previous-state features: x(t-1)."""
    X = pd.DataFrame(index=df.index)
    for c in BASE:
        X[f"{c}_lag1"] = df[c].shift(1)
    return X


def representation_b(df, windows=(6, 24), y=None, rain_mm=0.1):
    """Representation A + past-only temporal context."""
    X = representation_a(df)
    past = df[BASE].shift(1)  # value at row t is x(t-1)

    for c in BASE:
        X[f"{c}_delta1"] = past[c] - past[c].shift(1)  # x(t-1) - x(t-2)

    rainy = (past["precipitation"] >= rain_mm).astype(float).where(past["precipitation"].notna())
    for w in windows:
        for c in BASE:
            X[f"{c}_mean{w}"] = past[c].rolling(w).mean()
            X[f"{c}_std{w}"] = past[c].rolling(w).std()
        X[f"rain_hours_{w}"] = rainy.rolling(w).sum()
        if y is not None:  # labels up to t-1 are already revealed in the stream
            X[f"event_rate_{w}"] = y.shift(1).rolling(w).mean()
    return X
