"""The five mandatory incremental learners (fresh instances on every call)."""
from river import ensemble, forest, linear_model, naive_bayes, preprocessing, tree


def build_models(seed=42):
    return {
        "GaussianNB": naive_bayes.GaussianNB(),
        "OnlineLogReg": preprocessing.StandardScaler() | linear_model.LogisticRegression(),
        "HoeffdingTree": tree.HoeffdingTreeClassifier(),
        "OnlineBagging": ensemble.BaggingClassifier(
            model=tree.HoeffdingTreeClassifier(), n_models=10, seed=seed),
        "ARF": forest.ARFClassifier(n_models=10, seed=seed),
    }
