"""End-to-end pipeline runner to reproduce the notebook workflow."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import LinearSVC


try:
    from src.data_loading import load_datasets, RANDOM_STATE, DEFAULT_DATA_DIR
    from src.eda import (
        missing_table_direct,
        plot_class_imbalance,
        plot_usage_heatmap,
        plot_numeric_histograms,
        plot_numeric_correlation,
    )
    from src.timeseries import build_monthly_timeseries, plot_monthly_timeseries, evaluate_forecasting_baselines
    from src.preprocessing import (
        normalize_environment,
        join_datasets,
        drop_unnecessary_columns,
        encode_categoricals,
        split_train_test,
        detect_columns_by_type,
    )
    from src.imputation import get_imputation_strategies
    from src.modeling import evaluate_imputation, evaluate_multiclass_strategies
    from src.sampling import get_sampling_strategies, evaluate_sampling
except ImportError:
    from data_loading import load_datasets, RANDOM_STATE, DEFAULT_DATA_DIR
    from eda import (
        missing_table_direct,
        plot_class_imbalance,
        plot_usage_heatmap,
        plot_numeric_histograms,
        plot_numeric_correlation,
    )
    from timeseries import build_monthly_timeseries, plot_monthly_timeseries, evaluate_forecasting_baselines
    from preprocessing import (
        normalize_environment,
        join_datasets,
        drop_unnecessary_columns,
        encode_categoricals,
        split_train_test,
        detect_columns_by_type,
    )
    from imputation import get_imputation_strategies
    from modeling import evaluate_imputation, evaluate_multiclass_strategies
    from sampling import get_sampling_strategies, evaluate_sampling


def run_pipeline() -> None:
    """Run the full EDA + modeling pipeline using the refactored modules."""
    sns.set_theme(style="whitegrid")

    # Load datasets
    doors_registry, door_catalog, maintenance_history = load_datasets(DEFAULT_DATA_DIR)
    print("doors_registry:", doors_registry.shape)
    print("door_catalog:", door_catalog.shape)
    print("maintenance_history:", maintenance_history.shape)

    # Quick data overview
    print("\nFirst rows of each table:")
    print(doors_registry.head(3))
    print(maintenance_history.head(3))
    print(door_catalog.head(3))

    print("\nData types:")
    print("Door registry types of data:")
    print(doors_registry.dtypes)
    print("\nMaintenance history types of data:")
    print(maintenance_history.dtypes)
    print("\nDoor catalog types of data:")
    print(door_catalog.dtypes)

    missing_table_direct(doors_registry, "doors_registry")
    missing_table_direct(maintenance_history, "maintenance_history")
    missing_table_direct(door_catalog, "door_catalog")

    #  EDA visuals
    plot_class_imbalance(doors_registry, "door_type")
    plot_usage_heatmap(doors_registry, "door_type", "usage_scenario")

    # --- Merge for EDA ---
    doors_registry_norm = normalize_environment(doors_registry)
    eda_merged = join_datasets(doors_registry_norm, door_catalog, maintenance_history)

    eda_merged["last_maintenance_date"] = pd.to_datetime(
        eda_merged["last_maintenance_date"], errors="coerce"
    )
    snapshot_date = pd.Timestamp("2025-01-01")
    eda_merged["days_since_maintenance"] = (
        snapshot_date - eda_merged["last_maintenance_date"]
    ).dt.days

    numeric_cols = [
        "number_of_past_failures",
        "days_since_last_failure",
        "estimated_cycles_day",
        "days_since_maintenance",
    ]
    plot_numeric_histograms(eda_merged, numeric_cols)
    plot_numeric_correlation(eda_merged, numeric_cols)

    # Time series baseline
    monthly_ts = build_monthly_timeseries(maintenance_history)
    plot_monthly_timeseries(monthly_ts)
    evaluate_forecasting_baselines(monthly_ts["maintenance_events"].dropna())

    # Data transformation-
    final_dataset = drop_unnecessary_columns(
        eda_merged,
        [
            "door_id",
            "country_id",
            "days_to_next_failure",
            "failed_next_30_days",
            "last_maintenance_date",
        ],
    )

    # Dependency inspection (manual)
    target_col = "door_type"
    feature_df = final_dataset.drop(columns=[target_col], errors="ignore")
    num_cols, cat_cols = detect_columns_by_type(feature_df)
    print(f"Numerical cols: {len(num_cols)}")
    print(f"Categorical cols: {len(cat_cols)}")

    # Encode categoricals
    X = final_dataset.drop(columns=[target_col])
    y = final_dataset[target_col]
    X = encode_categoricals(X)

    X_train, X_test, y_train, y_test = split_train_test(X, y)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    # --- Imputation comparison ---
    imputation_strategies = get_imputation_strategies()
    imputation_results = evaluate_imputation(
        X_train, y_train, X_test, y_test, imputation_strategies
    )
    print("\nImputation results:")
    print(imputation_results)

    if not imputation_results.empty:
        best_imputer = imputation_results.sort_values("f1_score", ascending=False)["strategy"].iloc[0]
        print(f"Best imputation strategy: {best_imputer}")
        X_train, X_test = imputation_strategies[best_imputer](X_train, X_test)

    # Sampling comparison
    sampling_strategies = get_sampling_strategies()
    sampling_results = evaluate_sampling(X_train, y_train, X_test, y_test, sampling_strategies)
    print("\nSampling results:")
    print(sampling_results)

    if not sampling_results.empty:
        best_sampler = sampling_results.sort_values("f1_macro", ascending=False)["strategy"].iloc[0]
        print(f"Best sampling strategy: {best_sampler}")
        sampler = sampling_strategies.get(best_sampler)
        if sampler is not None and hasattr(sampler, "fit_resample"):
            X_train, y_train = sampler.fit_resample(X_train, y_train)

    # --- Multiclass strategies ---
    multiclass_models = {
        "Softmax": LogisticRegression(
            max_iter=3000,
            random_state=RANDOM_STATE,
            multi_class="multinomial",
        ),
        "One-vs-Rest": OneVsRestClassifier(LinearSVC(random_state=RANDOM_STATE)),
        "One-vs-One": OneVsOneClassifier(LinearSVC(random_state=RANDOM_STATE)),
    }

    multiclass_results = evaluate_multiclass_strategies(
        multiclass_models, X_train, y_train, X_test, y_test
    )
    print("\nMulticlass results:")
    print(multiclass_results)

    print("\nPipeline completed. Review plots and tables above.")



def main() -> None:

    run_pipeline()

if __name__ == "__main__":
    main()

