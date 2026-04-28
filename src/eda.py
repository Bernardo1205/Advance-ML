"""Exploratory data analysis helpers with clear, student-friendly visuals."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def missing_table_direct(df: pd.DataFrame, name: str) -> None:
    """Print a quick missing-value summary for each column."""
    print(f"\nMissing summary : {name}")
    print(f"{'Columna':<30} | {'Nulos':<10} | {'% Nulos':<10} | {'Únicos':<10}")
    for col in df.columns:
        missing_count = df[col].isna().sum()
        missing_pct = (df[col].isna().mean() * 100).round(2)
        unique_count = df[col].nunique(dropna=True)
        print(f"{col:<30} | {missing_count:<10} | {missing_pct:<10} | {unique_count:<10}")


def plot_class_imbalance(df: pd.DataFrame, target_col: str, save_path: str | None = None) -> float:
    """Plot a bar chart of class counts and return the imbalance ratio."""
    plt.figure(figsize=(10, 4))
    counts = df[target_col].value_counts()
    sns.barplot(x=counts.index, y=counts.values)
    plt.xticks(rotation=30, ha="right")
    plt.title(f"Class imbalance en {target_col}")
    plt.ylabel("count")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

    imbalance_ratio = counts.max() / max(counts.min(), 1)
    print(f"Imbalance ratio {target_col}: {round(imbalance_ratio, 2)}")
    return float(imbalance_ratio)


def plot_usage_heatmap(
    df: pd.DataFrame,
    target_col: str,
    usage_col: str,
    save_path: str | None = None,
) -> None:
    """Show the distribution of usage scenarios per door type."""
    # Group by door type and normalize per type for easy comparison.
    usage_table = df.groupby(target_col)[usage_col].value_counts(normalize=True).unstack(fill_value=0)
    plt.figure(figsize=(10, 5))
    sns.heatmap(usage_table, annot=True, cmap="Blues")
    plt.title(f"{target_col} vs {usage_col} (proportion per type)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_numeric_histograms(
    df: pd.DataFrame,
    numeric_cols: Iterable[str],
    save_path: str | None = None,
) -> None:
    """Plot histograms for the most relevant numeric variables."""
    df[list(numeric_cols)].hist(bins=20, figsize=(12, 8), color="orange", grid=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_numeric_correlation(
    df: pd.DataFrame,
    numeric_cols: Iterable[str],
    save_path: str | None = None,
) -> None:
    """Plot a correlation heatmap for numeric features."""
    plt.figure(figsize=(7, 5))
    corr = df[list(numeric_cols)].corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
    plt.title("Matriz de correlacion (variables numericas)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

