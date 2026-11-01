"""
MODELS
------

Define the five mandatory incremental learners and the optional reference
baselines, all built with `river`.

Notes:
- Models are state: Hoeffding trees grow their architecture as data arrives,
  while linear and NB models only update parameters.
- OnlineBagging uses Poisson resampling; ARF is an adaptive ensemble with
  per-tree drift detectors.

Builds on: course weeks W05 and W06.
"""
import math

from river import base, dummy, ensemble, forest, linear_model, naive_bayes, preprocessing, tree


def build_models(seed=42):
    return {
        "GaussianNB": naive_bayes.GaussianNB(),
        "OnlineLogReg": preprocessing.StandardScaler() | linear_model.LogisticRegression(),
        "HoeffdingTree": tree.HoeffdingTreeClassifier(),
        "OnlineBagging": ensemble.BaggingClassifier(
            model=tree.HoeffdingTreeClassifier(), n_models=10, seed=seed),
        "ARF": forest.ARFClassifier(n_models=10, seed=seed),
    }


def build_baselines(seed=42):
    """Return new reference models and whether they learn after warm-up."""
    return {
        "Majority": (dummy.PriorClassifier(), True),
        "Persistence": (dummy.NoChangeClassifier(), True),
        "FrozenLogReg": (
            preprocessing.StandardScaler() | linear_model.LogisticRegression(), False
        ),
    }


def model_complexity(model):
    """End-of-stream structural state; supporting evidence only (Guidelines 11.2, W05/W06).

    A common complexity measure across all algorithms is not mandatory, so models without
    trees (GaussianNB, LogReg) get NaN.
    """
    is_ensemble = isinstance(model, base.Ensemble)
    members = list(model) if is_ensemble else [model]
    trees = [m for m in members if getattr(m, "n_nodes", None) is not None]
    out = {
        "n_members": len(members) if is_ensemble else math.nan,
        "total_tree_nodes": sum(t.n_nodes for t in trees) if trees else math.nan,
        "mean_tree_height": (sum(t.height or 0 for t in trees) / len(trees)) if trees else math.nan,
        "drifts_detected": math.nan,
        "warnings_detected": math.nan,
    }
    if hasattr(model, "n_drifts_detected"):         
        out["drifts_detected"] = model.n_drifts_detected()
        out["warnings_detected"] = model.n_warnings_detected()
    return out
