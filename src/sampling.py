"""Sampling helpers for imbalanced classification."""

from __future__ import annotations
from typing import Dict
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

try:
    from src.data_loading import RANDOM_STATE
except ImportError:
    from data_loading import RANDOM_STATE


def get_sampling_strategies() -> Dict[str, object | None]:
    sampling_strategies = {
        "baseline": None,  # No sampling (comparison)
        "ENN (Under-sampling)": EditedNearestNeighbours(),
        "SMOTE (Over-sampling)": SMOTE(random_state=RANDOM_STATE),
        "SMOTEENN (Combined)": SMOTEENN(random_state=RANDOM_STATE),
    }

    return sampling_strategies


def evaluate_sampling(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    strategies: Dict[str, object | None],
) -> pd.DataFrame:
    """Train a RandomForest on each sampling strategy and compare metrics."""
    results = []

    for name, sampler in strategies.items():
        if sampler is None:
            X_res, y_res = X_train, y_train
        else:
            X_res, y_res = sampler.fit_resample(X_train, y_train)

        model = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
        model.fit(X_res, y_res)

        y_pred = model.predict(X_test)
        results.append({
            "strategy": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
        })

    return pd.DataFrame(results)
