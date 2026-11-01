"""
STREAM ANALYSIS
---------------

Characterise both weather streams and run the original analyses (O1-O4).
Writes `results/stream_summary.csv` and the stream-profile and comparison figures.

Same logic for both cities. Extremes are counted and dated, never removed.

Original analyses:
- O1  Race-window reliability, overall and by month.
- O2  Adverse/clear spell persistence, P(y_t = 1 | y_{t-1} = 1).
- O3  Hour-of-day and monthly event profiles, split by source (rain/wind/both).
- O4  Representation effect (Rep A vs B) vs processing cost.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_loader import quality_report

SOURCE_COLORS = {"rain only": "#2b7bba", "wind only": "#e6a532", "rain and wind": "#7a3e9d"}


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def label_frame(df, y, rain_mm, tau_w, local_tz):
    """One row per hour: label, event source and local time (labels only, no features)."""
    out = df[["time", "temperature", "relative_humidity", "precipitation", "wind_speed"]].copy()
    out["y"] = y.values
    rain = out["precipitation"] >= rain_mm
    wind = out["wind_speed"] >= tau_w
    out["source"] = np.select([rain & ~wind, wind & ~rain, rain & wind],
                              ["rain only", "wind only", "rain and wind"], default="None")
    out["local_time"] = (pd.to_datetime(out["time"]).dt.tz_localize("UTC")
                         .dt.tz_convert(local_tz).dt.tz_localize(None))
    return out


def spells(y):
    """Lengths (hours) of consecutive runs of y == 1 and y == 0."""
    y = pd.Series(y).dropna().astype(int).reset_index(drop=True)
    run_id = (y != y.shift()).cumsum()
    runs = y.groupby(run_id).agg(["first", "size"])
    return (runs.loc[runs["first"] == 1, "size"].to_numpy(),
            runs.loc[runs["first"] == 0, "size"].to_numpy())


def race_windows(lf, start_hour, n_hours):
    """Per local calendar day: True when every hour of the race window is non-adverse."""
    hours = lf["local_time"].dt.hour
    inside = lf[(hours >= start_hour) & (hours < start_hour + n_hours)]
    day = inside["local_time"].dt.normalize()
    grouped = inside.groupby(day)["y"].agg(["size", "max"])
    grouped = grouped[grouped["size"] == n_hours]          # complete windows only (DST days)
    return (grouped["max"] == 0).rename("reliable")


def city_summary(name, df, lf, warm_n, analysis_cfg):
    """Descriptive and event statistics for one city (same definitions for both cities)."""
    q = quality_report(df)
    y = lf["y"]
    events = lf[y == 1]
    adverse, clear = spells(y)
    persistence = ((y.shift(1) == 1) & (y == 1)).sum() / max((y.shift(1) == 1).sum(), 1)
    rolling24 = y.rolling(24).mean()
    rw = race_windows(lf, analysis_cfg["race_window"]["start_hour"],
                      analysis_cfg["race_window"]["n_hours"])
    monthly_rw = rw.groupby(rw.index.month).mean()
    daily_free = lf.groupby(lf["local_time"].dt.normalize())["y"].max().eq(0)
    wmax, pmax = lf.loc[lf["wind_speed"].idxmax()], lf.loc[lf["precipitation"].idxmax()]
    ext = analysis_cfg["extremes"]
    yearly = lf.groupby(lf["time"].dt.year)["y"].mean()
    return {
        "city": name,
        "n_obs": q["n_rows"], "start": q["start"], "end": q["end"],
        "missing_values": int(sum(q["missing"].values())),
        "duplicated_timestamps": q["duplicated_timestamps"],
        "irregular_steps": q["irregular_steps"],
        "invalid_values": q["humidity_out_of_range"] + q["negative_precipitation"] + q["negative_wind"],
        "warmup_event_rate": float(y.iloc[:warm_n].mean()),
        "event_rate": float(y.mean()),
        "rain_hour_rate": float((lf["precipitation"] >= analysis_cfg["rain_mm"]).mean()),
        "strong_wind_hour_rate": float((lf["wind_speed"] >= analysis_cfg["tau_w"]).mean()),
        "share_rain_only": float((events["source"] == "rain only").mean()),
        "share_wind_only": float((events["source"] == "wind only").mean()),
        "share_both": float((events["source"] == "rain and wind").mean()),
        "temperature_mean": float(lf["temperature"].mean()), "temperature_std": float(lf["temperature"].std()),
        "humidity_mean": float(lf["relative_humidity"].mean()),
        "wind_mean": float(lf["wind_speed"].mean()), "wind_std": float(lf["wind_speed"].std()),
        "wind_p95": float(lf["wind_speed"].quantile(0.95)),
        "precip_total_mm_per_year": float(lf["precipitation"].sum() / (q["n_rows"] / 8766)),
        "wind_max": float(wmax["wind_speed"]), "wind_max_time": str(wmax["time"]),
        "precip_max": float(pmax["precipitation"]), "precip_max_time": str(pmax["time"]),
        "hours_wind_ge_extreme": int((lf["wind_speed"] >= ext["wind_kmh"]).sum()),
        "hours_precip_ge_extreme": int((lf["precipitation"] >= ext["precip_mm"]).sum()),
        "persistence_p11": float(persistence),
        "adverse_spell_mean_h": float(adverse.mean()), "adverse_spell_median_h": float(np.median(adverse)),
        "adverse_spell_max_h": int(adverse.max()), "adverse_spells_ge_24h": int((adverse >= 24).sum()),
        "clear_spell_mean_h": float(clear.mean()), "clear_spell_median_h": float(np.median(clear)),
        "clear_spell_max_h": int(clear.max()),
        "rolling24_rate_std": float(rolling24.std()),
        "share_24h_windows_all_clear": float((rolling24 == 0).mean()),
        "share_24h_windows_all_adverse": float((rolling24 == 1).mean()),
        "event_free_day_share": float(daily_free.mean()),
        "race_window_reliability": float(rw.mean()), "race_window_days": int(rw.size),
        "race_window_best_month": int(monthly_rw.idxmax()), "race_window_best_value": float(monthly_rw.max()),
        "race_window_worst_month": int(monthly_rw.idxmin()), "race_window_worst_value": float(monthly_rw.min()),
        "yearly_event_rate_min": float(yearly.min()), "yearly_event_rate_max": float(yearly.max()),
    }


def plot_city_profile(name, lf, analysis_cfg, path):
    """Four-panel stream profile, identical layout for both cities."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), layout="constrained")
    ts = lf.set_index("time")
    roll = ts["y"].rolling(24 * 30, min_periods=24 * 30).mean()
    ax = axes[0, 0]
    ax.plot(roll.index, roll, color="black", lw=1)
    ax.set_title("(a) 30-day rolling T6 event rate")
    ax.set_ylabel("Share of adverse hours"); ax.set_ylim(0, 1); ax.grid(alpha=0.3)

    month = lf.groupby([lf["local_time"].dt.month, "source"]).size().unstack(fill_value=0)
    month = month.div(lf.groupby(lf["local_time"].dt.month).size(), axis=0)
    ax = axes[0, 1]
    bottom = np.zeros(12)
    for src, color in SOURCE_COLORS.items():
        vals = month.get(src, pd.Series(0, index=month.index)).reindex(range(1, 13)).fillna(0).to_numpy()
        ax.bar(range(1, 13), vals, bottom=bottom, color=color, label=src)
        bottom += vals
    ax.set_title("(b) Event rate by calendar month and source")
    ax.set_xticks(range(1, 13)); ax.set_xlabel("Month"); ax.set_ylim(0, 1); ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    hour = lf.groupby(lf["local_time"].dt.hour)["y"].mean()
    rw = analysis_cfg["race_window"]
    ax = axes[1, 0]
    ax.axvspan(rw["start_hour"] - 0.5, rw["start_hour"] + rw["n_hours"] - 0.5, color="#21A58B", alpha=0.15,
               label="race window")
    ax.plot(hour.index, hour.values, marker="o", color="black", ms=3)
    ax.set_title("(c) Event rate by local hour of day")
    ax.set_xlabel("Local hour"); ax.set_xticks(range(0, 24, 3)); ax.set_ylim(0, 1); ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    adverse, clear = spells(lf["y"])
    ax = axes[1, 1]
    bins = np.arange(1, 74, 3)
    ax.hist(np.clip(adverse, 1, 72), bins=bins, alpha=0.7, color="#c0392b", label="adverse spells")
    ax.hist(np.clip(clear, 1, 72), bins=bins, alpha=0.5, color="#21A58B", label="clear spells")
    ax.set_yscale("log"); ax.set_xlabel("Spell length (h, clipped at 72)"); ax.set_ylabel("Count")
    ax.set_title("(d) Duration of consecutive adverse / clear hours"); ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.suptitle(f"{name} - T6 stream profile (2021-01 to 2026-09)")
    _save(fig, path)


def plot_city_comparison(frames, analysis_cfg, path):
    """O1 + O2 side by side for both cities."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained")
    rw = analysis_cfg["race_window"]
    for name, lf in frames.items():
        windows = race_windows(lf, rw["start_hour"], rw["n_hours"])
        monthly = windows.groupby(windows.index.month).mean().reindex(range(1, 13))
        axes[0].plot(monthly.index, monthly.values, marker="o", label=name)
        adverse, clear = spells(lf["y"])
        for arr, ls, ax in ((clear, "-", axes[1]), (adverse, "--", axes[1])):
            grid = np.arange(1, 97)
            surv = [(arr >= g).mean() for g in grid]
            ax.plot(grid, surv, ls, label=f"{name} - {'clear' if arr is clear else 'adverse'}")
        yearly = lf.groupby(lf["time"].dt.year)["y"].mean()
        axes[2].plot(yearly.index, yearly.values, marker="s", label=name)
    axes[0].set_title(f"(a) Days with an adverse-free race window\n"
                      f"({rw['start_hour']:02d}:00-{rw['start_hour'] + rw['n_hours']:02d}:00 local)")
    axes[0].set_xlabel("Month"); axes[0].set_xticks(range(1, 13)); axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Share of days")
    axes[1].set_title("(b) Spell survival: P(spell lasts >= h hours)")
    axes[1].set_xlabel("Hours"); axes[1].set_yscale("log"); axes[1].set_ylim(1e-3, 1.05)
    axes[2].set_title("(c) T6 event rate by year"); axes[2].set_xlabel("Year (2026: Jan-Sep)")
    axes[2].set_ylim(0, 1); axes[2].set_ylabel("Share of adverse hours")
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    _save(fig, path)


def plot_model_evidence(metrics, path):
    """O4: Rep B - Rep A change in Macro-F1 and recall vs change in processing cost."""
    mand = metrics[metrics.role == "mandatory"].copy()
    mand["cost_ms"] = mand.pred_ms_per_event + mand.update_ms_per_event
    piv = mand.pivot_table(index=["city", "model"], columns="representation",
                           values=["macro_f1", "recall", "cost_ms"])
    delta = pd.DataFrame({
        "d_macro_f1": piv["macro_f1"]["B"] - piv["macro_f1"]["A"],
        "d_recall": piv["recall"]["B"] - piv["recall"]["A"],
        "cost_ratio": piv["cost_ms"]["B"] / piv["cost_ms"]["A"],
    }).reset_index()
    cities = list(dict.fromkeys(mand.city))
    models = list(dict.fromkeys(mand.model))
    fig, axes = plt.subplots(1, len(cities), figsize=(13, 4.3), sharey=True, layout="constrained")
    for ax, city in zip(np.atleast_1d(axes), cities):
        part = delta[delta.city == city].set_index("model").reindex(models)
        pos = np.arange(len(models))
        ax.bar(pos - 0.2, part.d_macro_f1, 0.4, label="Delta Macro-F1 (B - A)", color="#0B2F4F")
        ax.bar(pos + 0.2, part.d_recall, 0.4, label="Delta event recall (B - A)", color="#21A58B")
        for p, ratio in zip(pos, part.cost_ratio):
            ax.text(p, 0.97, f"cost x{ratio:.1f}", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8, color="#526575")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(pos, models, fontsize=9)
        ax.set_title(city); ax.grid(axis="y", alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("Change from Rep A to Rep B")
    np.atleast_1d(axes)[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Effect of temporal context (Rep B) on event detection, with cost multiplier")
    _save(fig, path)
    return delta


def run_stream_analysis(data, labels, names, thresholds, cfg, out_dir):
    """Write results/stream_summary.csv and the profile / comparison figures."""
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    acfg = dict(cfg["analysis"])
    acfg["rain_mm"] = cfg["targets"]["rain_threshold_mm"]
    acfg["tau_w"] = thresholds["T5"]
    frames, rows = {}, []
    for key, df in data.items():
        lf = label_frame(df, labels[key], acfg["rain_mm"], acfg["tau_w"], cfg["cities"][key]["local_timezone"])
        frames[names[key]] = lf
        rows.append(city_summary(names[key], df, lf, cfg["warmup"]["n_hours"], acfg))
        plot_city_profile(names[key], lf, acfg, fig_dir / f"{key}_stream_profile.png")
    plot_city_comparison(frames, acfg, fig_dir / "city_comparison.png")
    summary = pd.DataFrame(rows).set_index("city").T
    summary.to_csv(out_dir / "stream_summary.csv")
    return summary
