"""Official targets T1-T6, defined in config.yaml (`targets.definitions`).

Course alignment (Guidelines Ch. 6, 7 and 12):
- Data-derived thresholds are estimated ONCE from the warm-up rows of BOTH cities pooled,
  then frozen; they are never recomputed on evaluation data.
- Values observed at time t are used ONLY to build the label y_t, never as predictors at t.

Threshold keys returned by `compute_thresholds`:
- quantile target (T2-T5): ``{"T5": tau}``
- composite target (T6):   ``{"T6_rule_1": tau_W, "T5": tau_W}`` - the ``"T5"`` alias keeps the
  wind threshold readable under its official name (T6 = T1 OR T5).
"""
import operator

import pandas as pd

from data_loader import CONFIG

OPERATORS = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "==": operator.eq,
}

RAIN_MM = CONFIG["targets"].get("rain_threshold_mm", 0.1)

# Official quantile targets that a composite rule may reuse (variable, quantile) -> target id.
_OFFICIAL_QUANTILES = {("temperature", 0.80): "T2", ("temperature", 0.20): "T3",
                       ("relative_humidity", 0.50): "T4", ("wind_speed", 0.80): "T5"}


def _spec(target, targets_cfg):
    targets_cfg = targets_cfg or CONFIG["targets"]
    try:
        return targets_cfg["definitions"][target]
    except KeyError as exc:
        raise ValueError(f"Target {target} is not defined in config.yaml (targets.definitions)") from exc


def compute_thresholds(df_a, df_b, warmup_n, target, targets_cfg=None):
    """Estimate data-derived thresholds once from the first `warmup_n` rows of BOTH cities."""
    spec = _spec(target, targets_cfg)
    pooled = pd.concat([df_a.iloc[:warmup_n], df_b.iloc[:warmup_n]], ignore_index=True)
    thresholds = {}
    if spec["type"] == "quantile":
        thresholds[target] = float(pooled[spec["variable"]].quantile(spec["quantile"]))
    elif spec["type"] == "composite_or":
        for i, rule in enumerate(spec["rules"]):
            if rule["type"] == "quantile":
                tau = float(pooled[rule["variable"]].quantile(rule["quantile"]))
                thresholds[f"{target}_rule_{i}"] = tau
                alias = _OFFICIAL_QUANTILES.get((rule["variable"], rule["quantile"]))
                if alias:
                    thresholds[alias] = tau
    return thresholds  # frozen from here on


def make_target(df, target, thresholds, targets_cfg=None):
    """Binary label y_t from values observed at time t (used ONLY as the label)."""
    spec = _spec(target, targets_cfg)
    kind = spec["type"]

    if kind == "threshold":
        src = [spec["variable"]]
        y = OPERATORS[spec["operator"]](df[spec["variable"]], spec["threshold"])
    elif kind == "quantile":
        src = [spec["variable"]]
        y = OPERATORS[spec["operator"]](df[spec["variable"]], thresholds[target])
    elif kind == "composite_or":
        src, conditions = [], []
        for i, rule in enumerate(spec["rules"]):
            if rule["type"] == "threshold":
                value = rule["threshold"]
            elif rule["type"] == "quantile":
                value = thresholds[f"{target}_rule_{i}"]
            else:
                raise ValueError(f"Unknown rule type: {rule.get('type')}")
            src.append(rule["variable"])
            conditions.append(OPERATORS[rule["operator"]](df[rule["variable"]], value))
        y = conditions[0]
        for condition in conditions[1:]:
            y = y | condition
    else:
        raise ValueError(f"Unknown target definition type: {kind}")

    y = y.astype(float).where(df[src].notna().all(axis=1))  # keep missing as NaN
    return y.rename("y")

