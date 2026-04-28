"""Model evaluation helpers for imputation and multiclass strategies."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

try:
    from src.data_loading import RANDOM_STATE
except ImportError:
    from data_loading import RANDOM_STATE


def evaluate_imputation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    strategies: dict,
) -> pd.DataFrame:
    """Evaluate multiple imputation strategies with a fixed classifier."""
    results = []
    for name, imputer_fn in strategies.items():
        # Impute only the feature matrices (never touch labels).
        X_tr_imp, X_te_imp = imputer_fn(X_train, X_test)

        model = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
        model.fit(X_tr_imp, y_train)

        y_pred = model.predict(X_te_imp)
        results.append({
            "strategy": name,
            "f1_score": f1_score(y_test, y_pred, average="macro"),
        })

    return pd.DataFrame(results)


def evaluate_multiclass_strategies(
    models: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Compare multiclass classifiers on a single metric (accuracy)."""
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rows.append({"strategy": name, "accuracy": accuracy_score(y_test, y_pred)})

    return pd.DataFrame(rows)
