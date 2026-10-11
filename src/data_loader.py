"""Retrieve hourly Open-Meteo history and run basic data-quality checks."""
from pathlib import Path

import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]          # repo root (src/ -> ..)


def load_config(path=ROOT / "config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG = load_config()


def fetch_city(name, latitude, longitude, start_date, end_date,
               timezone="UTC", raw_dir=None, use_cache=True, config=CONFIG):
    """Download (or load the cached snapshot of) one city's hourly stream."""
    if latitude is None or longitude is None:
        raise ValueError(f"Fill in latitude/longitude for {name} in config.yaml")

    raw_dir = Path(raw_dir) if raw_dir else ROOT / config["paths"]["raw_data"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{name.lower().replace(' ', '_')}_{start_date}_{end_date}.csv"

    if use_cache and path.exists():
        return pd.read_csv(path, parse_dates=["time"])

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(config["hourly_vars"]),   # joins the API names (dict keys)
        "timezone": timezone,
    }
    response = requests.get(config["paths"]["data_url"], params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()

    df = pd.DataFrame(payload["hourly"]).rename(columns=config["hourly_vars"])
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