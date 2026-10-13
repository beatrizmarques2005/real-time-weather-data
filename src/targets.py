'''"""Official targets T1-T6. Thresholds come from POOLED WARM-UP data only and are frozen.

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

'''

import operator
import pandas as pd

OPERATORS = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "==": operator.eq,
}

RAIN_MM = 0.1  # retained for backwards compatibility with imports in notebooks

def compute_thresholds(df_a, df_b, warmup_n, target, target_cfg=None):
    """Estimate quantile thresholds from the first warmup_n rows of BOTH cities."""
    pooled = pd.concat([df_a.iloc[:warmup_n], df_b.iloc[:warmup_n]], ignore_index=True)
    thresholds = {}

    # If target_cfg is provided, read dynamically; otherwise use fallback
    if target_cfg and target in target_cfg.get("definitions", {}):
        spec = target_cfg["definitions"][target]
        if spec["type"] == "quantile":
            thresholds[target] = float(pooled[spec["variable"]].quantile(spec["quantile"]))
        elif spec["type"] == "composite_or":
            for i, rule in enumerate(spec["rules"]):
                if rule["type"] == "quantile":
                    thresholds[f"{target}_rule_{i}"] = float(pooled[rule["variable"]].quantile(rule["quantile"]))
    else:
        # Fallback to standard rubric targets (preserves original behavior)
        if target in ["T2", "T3", "T4", "T5"]:
            quantiles = {"T2": ("temperature", 0.80), "T3": ("temperature", 0.20),
                         "T4": ("relative_humidity", 0.50), "T5": ("wind_speed", 0.80)}
            var, q = quantiles[target]
            thresholds[target] = float(pooled[var].quantile(q))
        elif target == "T6":
            thresholds["T5"] = float(pooled["wind_speed"].quantile(0.80))

    return thresholds


def make_target(df, target, thresholds, targets_cfg):
    """Compute binary label y_t from values observed at time t (used ONLY as the label)."""
    spec = targets_cfg["definitions"][target]
    target_type = spec.get("type")

    if target_type == "threshold":
        op = OPERATORS[spec["operator"]]
        src = [spec["variable"]]
        y = op(df[spec["variable"]], spec["threshold"])

    elif target_type == "quantile":
        op = OPERATORS[spec["operator"]]
        src = [spec["variable"]]
        tau = thresholds[target]
        y = op(df[spec["variable"]], tau)

    elif target_type == "composite_or":
        src = []
        conditions = []
        for i, rule in enumerate(spec["rules"]):
            var = rule["variable"]
            src.append(var)
            op = OPERATORS[rule["operator"]]

            if rule["type"] == "threshold":
                val = rule["threshold"]
            elif rule["type"] == "quantile":
                val = thresholds[f"{target}_rule_{i}"]
            else:
                raise ValueError(f"Unknown rule type: {rule.get('type')}")

            conditions.append(op(df[var], val))

        # Chain rules with logical OR
        y = conditions[0]
        for cond in conditions[1:]:
            y = y | cond
    else:
        raise ValueError(f"Unknown target definition type: {target_type}")

    # Keep missing measurements as NaN
    y = y.astype(float).where(df[src].notna().all(axis=1))
    return y.rename("y")