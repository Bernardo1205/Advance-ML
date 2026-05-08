"""Time series aggregation and baseline forecasting utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


def build_monthly_timeseries(history_df: pd.DataFrame) -> pd.DataFrame:
    """Create monthly counts and averages for maintenance events."""
    ts_history = history_df[["last_maintenance_date", "number_of_past_failures"]].copy()
    ts_history["last_maintenance_date"] = pd.to_datetime(ts_history["last_maintenance_date"], errors="coerce")
    ts_history = ts_history.dropna(subset=["last_maintenance_date"]).sort_values("last_maintenance_date")

    monthly_events = (
        ts_history.set_index("last_maintenance_date")
        .resample("MS")
        .size()
        .rename("maintenance_events")
    )
    monthly_failures_mean = (
        ts_history.set_index("last_maintenance_date")["number_of_past_failures"]
        .resample("MS")
        .mean()
        .rename("avg_past_failures")
    )

    monthly_ts = pd.concat([monthly_events, monthly_failures_mean], axis=1)
    return monthly_ts


def plot_monthly_timeseries(monthly_ts: pd.DataFrame, save_path: str | None = None) -> None:
    """Plot monthly maintenance events and average failures side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    monthly_ts["maintenance_events"].plot(ax=axes[0], marker="o", color="blue")
    axes[0].set_title("Monthly maintenance events")
    axes[0].set_ylabel("count")

    monthly_ts["avg_past_failures"].plot(ax=axes[1], marker="o", color="red")
    axes[1].set_title("Monthly mean of past failures")
    axes[1].set_ylabel("mean")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def evaluate_forecasting_baselines(
    series: pd.Series,
    save_path: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Compare naive vs rolling-mean forecasts for a monthly series."""
    metrics: dict = {}

    if len(series) < 8:
        print("La serie temporal es demasiado corta (menos de 8 meses) para evaluar métricas.")
        return pd.DataFrame(), metrics

    # Temporal split without shuffling.
    train_ts, test_ts = train_test_split(series, test_size=0.2, shuffle=False)

    # Naive baseline: last seen value.
    naive_pred = pd.Series(train_ts.iloc[-1], index=test_ts.index)

    # Rolling mean baseline: average of the last 3 months of training.
    rolling_pred = pd.Series(train_ts.tail(3).mean(), index=test_ts.index)

    naive_mae = mean_absolute_error(test_ts, naive_pred)
    naive_rmse = float(np.sqrt(mean_squared_error(test_ts, naive_pred)))

    rolling_mae = mean_absolute_error(test_ts, rolling_pred)
    rolling_rmse = float(np.sqrt(mean_squared_error(test_ts, rolling_pred)))

    results = pd.DataFrame([
        {"model": "naive_last_value", "mae": naive_mae, "rmse": naive_rmse},
        {"model": "rolling_mean_3", "mae": rolling_mae, "rmse": rolling_rmse},
    ]).sort_values("mae")

    best = results.iloc[0]
    metrics = {
        "best_model": best["model"],
        "mae": float(best["mae"]),
        "rmse": float(best["rmse"]),
        "n_train": int(len(train_ts)),
        "n_test": int(len(test_ts)),
    }

    # Plot comparison for interpretability.
    plt.figure(figsize=(10, 4))
    plt.plot(train_ts.index, train_ts.values, label="train", color="blue")
    plt.plot(test_ts.index, test_ts.values, label="test_real", color="green", marker="o")
    plt.plot(test_ts.index, naive_pred.values, label="naive_forecast", linestyle="--", color="orange")
    plt.plot(test_ts.index, rolling_pred.values, label="rolling_forecast", linestyle="--", color="red")
    plt.title("Comparativa de pronosticos vs mantenimientos reales")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

    return results, metrics

