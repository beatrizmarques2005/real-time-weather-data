"""The five mandatory incremental learners and optional reference baselines.

Course alignment:
- Guidelines Ch. 10: the five mandatory incremental learners are kept in build_models().
- Guidelines 10.1: build_baselines() adds optional reference learners; they do NOT replace the five.
- W05: models are state; Hoeffding trees grow architecture, linear/NB models only update parameters.
- W06: OnlineBagging (Poisson resampling) and ARF (adaptive ensemble with per-tree drift detectors).
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
        # W04 "Better than what?": online majority-class floor.
        "Majority": (dummy.PriorClassifier(), True),
        # Naive temporal floor: predicts the last revealed label y(t-1).
        "Persistence": (dummy.NoChangeClassifier(), True),
        # W04 prediction != online learning: same pipeline as OnlineLogReg,
        # learns on warm-up only, then predicts without updating.
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
    # A tree that has just been reset (e.g. an ARF member replaced after a drift) may report
    # n_nodes/height as None; such members are skipped instead of crashing the summary.
    trees = [m for m in members if getattr(m, "n_nodes", None) is not None]
    out = {
        "n_members": len(members) if is_ensemble else math.nan,
        "total_tree_nodes": sum(t.n_nodes for t in trees) if trees else math.nan,
        "mean_tree_height": (sum(t.height or 0 for t in trees) / len(trees)) if trees else math.nan,
        "drifts_detected": math.nan,
        "warnings_detected": math.nan,
    }
    if hasattr(model, "n_drifts_detected"):          # ARF only (ADWIN per tree by default)
        out["drifts_detected"] = model.n_drifts_detected()
        out["warnings_detected"] = model.n_warnings_detected()
    return out
