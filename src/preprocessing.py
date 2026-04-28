"""Data cleaning, joins, encoding, and train/test split helpers."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from src.data_loading import RANDOM_STATE
except ImportError:
    from data_loading import RANDOM_STATE


def normalize_environment(doors_registry: pd.DataFrame) -> pd.DataFrame:
    """Normalize environment labels so joins do not drop rows."""
    doors_registry = doors_registry.copy()
    doors_registry["installation_environment"] = doors_registry["installation_environment"].replace(
        "Community", "Residential Community"
    )
    return doors_registry


def join_datasets(
    doors_registry: pd.DataFrame,
    door_catalog: pd.DataFrame,
    maintenance_history: pd.DataFrame,
) -> pd.DataFrame:
    """Join registry, history, and catalog into a single feature table.

    Join plan:
    1) `doors_registry` LEFT JOIN `maintenance_history` on door_id to keep every door.
    2) The result LEFT JOIN `door_catalog` on door_type, usage_scenario, and installation_environment.
       These three columns are the shared business keys across the datasets.
    """
    # Join 1: keep all doors and attach maintenance info when present.
    merged = doors_registry.merge(maintenance_history, on="door_id", how="left")

    # Join 2: enrich with catalog metadata using the shared categorical keys.
    merged = merged.merge(
        door_catalog,
        on=["door_type", "usage_scenario", "installation_environment"],
        how="left",
    )
    return merged


def drop_unnecessary_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Drop columns that are not used for modeling or have 100% missing values."""
    return df.drop(columns=list(columns), errors="ignore")


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split for multiclass targets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def encode_categoricals(X: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode all non-numeric features using a simple pandas approach."""
    categorical_cols = X.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=int)
    return X_encoded


def detect_columns_by_type(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return numeric and categorical column names for quick inspection."""
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()
    return num_cols, cat_cols

