import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import NUMERIC_FEATURES, NUMERIC_COLS


def plot_status_distribution(df: pd.DataFrame) -> None:
    """Plot the count distribution of the target variable 'Status'."""
    status_counts = df["Status"].value_counts()
    status_counts.plot(kind="bar")
    plt.title("Count by Status")
    plt.xlabel("Status")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
    print("Status patient distribution")
    return status_counts


def plot_missing_values(df: pd.DataFrame) -> pd.Series:
    """Plot and report missing values per column."""
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=True)

    if len(missing) > 0:
        barras = plt.barh(missing.index, missing.values)
        plt.xlabel("Missing values (count)")
        plt.title("Missing values by column")

        etiquetas = [f"{v} ({(v / len(df)) * 100:.1f}%)" for v in missing.values]
        plt.bar_label(barras, labels=etiquetas, padding=3, fontsize=8)
        plt.tight_layout()
        plt.show()

    print("\n[Missing values]")
    if len(missing) > 0:
        print("Columns with missing data (may require imputation):")
        print(missing.to_string())
    else:
        print("No missing values found.")

    return missing


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    """Plot histograms for key numeric features."""
    print("\n[Numeric distributions]")
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            plt.hist(df[col].dropna(), bins=30)
            plt.title(col)
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.show()
            print(f"{col}: shows general shape (skew/spread).")


def plot_correlation_matrix(df: pd.DataFrame) -> None:
    """Plot a lower-triangle correlation heatmap for numeric columns."""
    cols = [c for c in NUMERIC_COLS if c in df.columns]

    if len(cols) > 0:
        corr_matrix = df[cols].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
                    cmap="coolwarm", center=0)
        plt.title("Correlation matrix (numeric)")
        plt.tight_layout()
        plt.show()

        print("\n[Correlation]")
        print("Values close to 1/-1 indicate strong relation; close to 0 indicate weak relation.")


def plot_stage_analysis(df: pd.DataFrame) -> None:
    """Plot case counts and Status distribution within each Stage."""
    if "Stage" not in df.columns:
        return

    stage_counts = df["Stage"].value_counts().sort_index()
    plt.bar(stage_counts.index.astype(str), stage_counts.values)
    plt.xlabel("Stage")
    plt.ylabel("Count")
    plt.title("Cases by Stage")
    plt.tight_layout()
    plt.show()
    print("\n[Stage] Count per stage shown above.")

    stage_status = pd.crosstab(df["Stage"], df["Status"], normalize="index") * 100
    stage_status.plot(kind="bar", stacked=True)
    plt.xlabel("Stage")
    plt.ylabel("%")
    plt.title("Status % distribution within each Stage")
    plt.tight_layout()
    plt.show()
    print("Status percentage within each Stage shown above.")


def run_eda(df: pd.DataFrame) -> None:
    """Run the full EDA pipeline."""
    print(df.shape)
    print(df.columns.tolist())
    plt.style.use("default")

    status_counts = plot_status_distribution(df)
    missing = plot_missing_values(df)
    plot_numeric_distributions(df)
    plot_correlation_matrix(df)
    plot_stage_analysis(df)

    numeric_cols = [c for c in NUMERIC_COLS if c in df.columns]
    print("\nQuick summary")
    print("Status counts:")
    print(status_counts.to_string())
    print("Missing (if any):")
    print(missing.to_string())
    print("Numeric description (selected columns):")
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe().round(2).to_string())