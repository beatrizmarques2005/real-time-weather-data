"""
FEATURES
--------

Build leakage-free feature representations for the prequential loop.
Every feature uses only observations available up to t-1.

Builds on: course week W03.

Representations:
- A (previous state): x(t-1) for each weather variable.
- B (stream-aware context): A plus past-only deltas, rolling mean and std,
  rainy-hour counts, and the rate of already-revealed past adverse events.

Rule:
- precipitation(t) and wind_speed(t) define the target, so they are never
  used as predictors at time t.

"""

import pandas as pd

BASE = ["temperature", 
        "relative_humidity", 
        "precipitation", 
        "wind_speed"]


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
        if y is not None:
            X[f"event_rate_{w}"] = y.shift(1).rolling(w).mean()
    return X
