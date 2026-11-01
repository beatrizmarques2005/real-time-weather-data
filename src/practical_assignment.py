"""
PRACTICAL ASSIGNMENT
--------------------

Main experiment: 2 cities x 2 representations x 5 mandatory models
(+ optional reference baselines), evaluated with the prequential protocol
(PREDICT -> EVALUATE -> LEARN).

Notes:
- Chronological replay only; no shuffling and no random train/test split.
- Shared warm-up: the first `warmup.n_hours` rows fix the pooled frozen
  threshold and are learned but not scored, with an identical warm-up mask
  for every learner.
- Identical evaluated rows (shared `valid` mask) for every model and both
  representations.
- A fresh model instance for every run, with a fixed seed.

Outputs (audit package):
- metrics.csv, history.csv.gz and configurations.csv (package versions and
  config), written to `results/`.

Usage:
    python src/practical_assignment.py --config config.yaml

Builds on: course week W04.
"""
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

from data_loader import ROOT, fetch_city, quality_report
from features import representation_a, representation_b
from models import build_baselines, build_models, model_complexity
from plots import generate_plots
from prequential import run_prequential
from stream_analysis import plot_model_evidence, run_stream_analysis
from targets import compute_thresholds, make_target


def main(config_path):
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    target = cfg["assignment"]["target"]
    period, warm_n = cfg["period"], cfg["warmup"]["n_hours"]
    window_size = cfg["evaluation"]["window_size"]
    track_memory = cfg["evaluation"]["track_memory"]
    include_baselines = cfg["baselines"]["include"]
    out = ROOT / cfg["paths"]["results"]
    out.mkdir(parents=True, exist_ok=True)

    # 1. Data
    data = {}
    for key in ("city_a", "city_b"):
        c = cfg["cities"][key]
        df = fetch_city(c["name"], c["latitude"], c["longitude"], period["start_date"],
                        period["end_date"], period["timezone"], ROOT / cfg["paths"]["raw_data"])
        print(f"[{c['name']}] quality:", quality_report(df), flush=True)
        data[key] = df

    # 2. Frozen thresholds from pooled warm-up
    # Frozen once, before any evaluation
    thresholds = compute_thresholds(data["city_a"], data["city_b"], warm_n, target, cfg["targets"])
    print("Frozen thresholds:", thresholds, flush=True)
    labels = {key: make_target(df, target, thresholds, cfg["targets"]) for key, df in data.items()}
    names = {key: cfg["cities"][key]["name"] for key in data}

    # 2b. Stream characterisation + original analyses (descriptive only)
    summary = run_stream_analysis(data, labels, names, thresholds, cfg, out)
    print("Stream summary saved:", out / "stream_summary.csv", flush=True)
    print(summary.loc[["event_rate", "race_window_reliability", "persistence_p11",
                       "adverse_spell_mean_h"]].to_string(), flush=True)

    # 3. Identical evaluated rows for every learner and representation.
    rows, histories = [], []
    for key, df in data.items():
        city = cfg["cities"][key]["name"]
        y = labels[key]
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
                             "model": model_name, "role": role, **res,
                             **model_complexity(model)})
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
        "window_size": window_size, "features_windows": cfg["features"]["windows"],
        "race_window": cfg["analysis"]["race_window"], "platform": platform.platform(), "track_memory": track_memory,
        "include_baselines": include_baselines,
        "python": platform.python_version(), "river": river.__version__,
        "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
        "pandas": pd.__version__, "numpy": np.__version__,
        **{f"model_{k}": str(v._get_params()) for k, v in build_models(seed).items()},
        **{f"baseline_{k}": f"{v._get_params()}; online={online}"
           for k, (v, online) in build_baselines(seed).items()},
    }
    pd.Series(config_rows).to_csv(out / "configurations.csv", header=["value"])
    generate_plots(history_df, metrics_df, out / "figures")
    plot_model_evidence(metrics_df, out / "figures" / "model_evidence.png")
    print(f"Saved results to {out}/", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    main(parser.parse_args().config)
