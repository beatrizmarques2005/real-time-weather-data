"""The five mandatory incremental learners and optional reference baselines."""
from river import dummy, ensemble, forest, linear_model, naive_bayes, preprocessing, tree


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
