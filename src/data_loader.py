"""Retrieve hourly Open-Meteo history and run basic data-quality checks."""
from pathlib import Path

import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = {
    "temperature_2m": "temperature",
    "relative_humidity_2m": "relative_humidity",
    "precipitation": "precipitation",
    "wind_speed_10m": "wind_speed",
}


def fetch_city(name, latitude, longitude, start_date, end_date,
               timezone="UTC", raw_dir="data/raw", use_cache=True):
    """Download (or load the cached snapshot of) one city's hourly stream."""
    if latitude is None or longitude is None:
        raise ValueError(f"Fill in latitude/longitude for {name} in config.yaml")
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{name.lower().replace(' ', '_')}_{start_date}_{end_date}.csv"

    if use_cache and path.exists():
        return pd.read_csv(path, parse_dates=["time"])

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": timezone,
    }
    response = requests.get(ARCHIVE_URL, params=params, timeout=60)
    response.raise_for_status()
    df = pd.DataFrame(response.json()["hourly"]).rename(columns=HOURLY_VARS)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    df.to_csv(path, index=False)
    return df


def quality_report(df):
    """Missing values, duplicates, gaps and physically invalid values.
    Extreme values are NOT treated as errors - only report them."""
    steps = df["time"].diff().dropna()
    return {
        "n_rows": len(df),
        "start": str(df["time"].min()),
        "end": str(df["time"].max()),
        "missing": df.isna().sum().to_dict(),
        "duplicated_timestamps": int(df["time"].duplicated().sum()),
        "irregular_steps": int((steps != pd.Timedelta(hours=1)).sum()),
        "humidity_out_of_range": int(((df["relative_humidity"] < 0) | (df["relative_humidity"] > 100)).sum()),
        "negative_precipitation": int((df["precipitation"] < 0).sum()),
        "negative_wind": int((df["wind_speed"] < 0).sum()),
    }
