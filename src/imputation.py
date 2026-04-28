"""Imputation helpers for numeric feature matrices."""

from __future__ import annotations

import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer

try:
    from src.data_loading import RANDOM_STATE
except ImportError:
    from data_loading import RANDOM_STATE

from missforest import MissForest



def mic_imputation(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """MICE (IterativeImputer) for multivariate imputation."""
    imputer = IterativeImputer(random_state=RANDOM_STATE)
    train_imp = pd.DataFrame(imputer.fit_transform(train), columns=train.columns, index=train.index)
    test_imp = pd.DataFrame(imputer.transform(test), columns=test.columns, index=test.index)
    return train_imp, test_imp


def forest_mic_imputation(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """MissForest for multivariate imputation using Random Forests."""
    imputer = MissForest()
    train_imp = pd.DataFrame(imputer.fit_transform(train), columns=train.columns, index=train.index)
    test_imp = pd.DataFrame(imputer.transform(test), columns=test.columns, index=test.index)
    return train_imp, test_imp


def knn_imputation(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """KNN-based imputation for numeric features."""
    imputer = KNNImputer()
    train_imp = pd.DataFrame(imputer.fit_transform(train), columns=train.columns, index=train.index)
    test_imp = pd.DataFrame(imputer.transform(test), columns=test.columns, index=test.index)
    return train_imp, test_imp


def get_imputation_strategies() -> dict:
    """Return the imputation strategies available in the current environment."""
    strategies = {
        "MICE": mic_imputation,
        "KNN": knn_imputation,
        "MissForest": forest_mic_imputation
    }
    return strategies

