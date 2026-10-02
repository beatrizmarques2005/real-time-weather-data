"""Main experiment: 2 cities x 2 representations x 5 models, prequential protocol."""
import argparse
import platform
import random
from pathlib import Path

import numpy as np
import pandas as pd
import river
import yaml

from data_loader import fetch_city, quality_report
from features import representation_a, representation_b
from models import build_models
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
    out = Path(cfg["paths"]["results"])
    out.mkdir(parents=True, exist_ok=True)

    # 1. Data
    data = {}
    for key in ("city_a", "city_b"):
        c = cfg["cities"][key]
        df = fetch_city(c["name"], c["latitude"], c["longitude"], period["start_date"],
                        period["end_date"], period["timezone"], cfg["paths"]["raw_data"])
        print(f"[{c['name']}] quality:", quality_report(df))
        data[key] = df

    # 2. Frozen thresholds from pooled warm-up
    thresholds = compute_thresholds(data["city_a"], data["city_b"], warm_n, target)
    print("Frozen thresholds:", thresholds)

    # 3. Experiment
    rows = []
    for key, df in data.items():
        city = cfg["cities"][key]["name"]
        y = make_target(df, target, thresholds)
        reps = {"A": representation_a(df),
                "B": representation_b(df, cfg["features"]["windows"], y=y)}
        # same rows for both representations -> same evaluation history
        valid = reps["B"].notna().all(axis=1) & y.notna()
        is_warm = pd.Series(df.index < warm_n, index=df.index)
        for rep_name, X in reps.items():
            for model_name, model in build_models(seed).items():  # fresh instances
                res = run_prequential(model, X[valid], y[valid].astype(int), is_warm[valid])
                rows.append({"city": city, "representation": rep_name, "model": model_name, **res})
                print(f"{city} | Rep {rep_name} | {model_name}: "
                      f"MacroF1={res['macro_f1']:.3f} Recall={res['recall']:.3f}")

    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)

    # 4. Configuration record for auditability
    config_rows = {
        "assignment_code": cfg["assignment"]["code"], "target": target,
        "city_a": cfg["cities"]["city_a"]["name"], "city_b": cfg["cities"]["city_b"]["name"],
        "start_date": period["start_date"], "end_date": period["end_date"],
        "warmup_hours": warm_n, "seed": seed, "thresholds": thresholds,
        "python": platform.python_version(), "river": river.__version__,
        "pandas": pd.__version__, "numpy": np.__version__,
        **{f"model_{k}": repr(v) for k, v in build_models(seed).items()},
    }
    pd.Series(config_rows).to_csv(out / "configurations.csv", header=["value"])
    print(f"Saved results to {out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    main(parser.parse_args().config)
