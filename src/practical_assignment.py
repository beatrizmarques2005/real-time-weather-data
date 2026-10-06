"""Main experiment: 2 cities x 2 representations x 5 mandatory models (+ baselines)."""
import argparse
import platform
import random
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import river
import sklearn
import yaml

from data_loader import fetch_city, quality_report
from features import representation_a, representation_b
from models import build_baselines, build_models
from plots import generate_plots
from prequential import run_prequential
from targets import compute_thresholds, make_target


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    target = cfg["assignment"]["target"]
    period, warm_n = cfg["period"], cfg["warmup"]["n_hours"]
    window_size = cfg["evaluation"]["window_size"]
    track_memory = cfg["evaluation"]["track_memory"]
    include_baselines = cfg["baselines"]["include"]
    out = Path(cfg["paths"]["results"])
    out.mkdir(parents=True, exist_ok=True)

    # 1. Data
    data = {}
    for key in ("city_a", "city_b"):
        c = cfg["cities"][key]
        df = fetch_city(c["name"], c["latitude"], c["longitude"], period["start_date"],
                        period["end_date"], period["timezone"], cfg["paths"]["raw_data"])
        print(f"[{c['name']}] quality:", quality_report(df), flush=True)
        data[key] = df

    # 2. Frozen thresholds from pooled warm-up
    thresholds = compute_thresholds(data["city_a"], data["city_b"], warm_n, target)
    print("Frozen thresholds:", thresholds, flush=True)

    # 3. Identical evaluated rows for every learner and representation.
    rows, histories = [], []
    for key, df in data.items():
        city = cfg["cities"][key]["name"]
        y = make_target(df, target, thresholds)
        reps = {"A": representation_a(df),
                "B": representation_b(df, cfg["features"]["windows"], y=y)}
        valid = reps["B"].notna().all(axis=1) & y.notna()
        is_warm = pd.Series(df.index < warm_n, index=df.index)
        for rep_name, X in reps.items():
            learners = [(name, model, True, "mandatory")
                        for name, model in build_models(seed).items()]
            if include_baselines:
                learners += [(name, model, online, "baseline")
                             for name, (model, online) in build_baselines(seed).items()]
            for model_name, model, online, role in learners:
                res, history = run_prequential(
                    model, X[valid], y[valid].astype(int), is_warm[valid], online=online,
                    window_size=window_size, timestamps=df.loc[valid, "time"].tolist(),
                    track_memory=track_memory)
                rows.append({"city": city, "representation": rep_name,
                             "model": model_name, "role": role, **res})
                history = history.assign(city=city, representation=rep_name,
                                         model=model_name, role=role)
                histories.append(history)
                print(f"{city} | Rep {rep_name} | {model_name}: "
                      f"MacroF1={res['macro_f1']:.3f} Recall={res['recall']:.3f} "
                      f"WindowMacroF1={res['window_macro_f1_final']:.3f} "
                      f"cold_starts={res['cold_starts']} checks OK", flush=True)

    metrics_df = pd.DataFrame(rows)
    history_df = pd.concat(histories, ignore_index=True)
    metrics_df.to_csv(out / "metrics.csv", index=False)
    history_df.to_csv(out / "history.csv.gz", index=False, compression="gzip")

    # 4. Configuration record for auditability
    config_rows = {
        "assignment_code": cfg["assignment"]["code"], "target": target,
        "city_a": cfg["cities"]["city_a"]["name"], "city_b": cfg["cities"]["city_b"]["name"],
        "start_date": period["start_date"], "end_date": period["end_date"],
        "warmup_hours": warm_n, "seed": seed, "thresholds": thresholds,
        "window_size": window_size, "track_memory": track_memory,
        "include_baselines": include_baselines,
        "python": platform.python_version(), "river": river.__version__,
        "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
        "pandas": pd.__version__, "numpy": np.__version__,
        **{f"model_{k}": repr(v) for k, v in build_models(seed).items()},
        **{f"baseline_{k}": f"{repr(v)}; online={online}"
           for k, (v, online) in build_baselines(seed).items()},
    }
    pd.Series(config_rows).to_csv(out / "configurations.csv", header=["value"])
    generate_plots(history_df, metrics_df, out / "figures")
    print(f"Saved results to {out}/", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    main(parser.parse_args().config)
