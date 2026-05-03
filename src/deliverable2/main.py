import warnings

import pandas as pd
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")

from config import DATA_PATH
from eda import run_eda
from preprocessing import (
    apply_log_transform,
    encode_features,
    evaluate_multiclass_strategies,
    select_best_imputation,
    select_best_sampling,
    split_data,
)
from explainability import (
    encode_labels,
    plot_ale,
    plot_pdp,
    plot_shap_summary,
    plot_shap_waterfall,
    run_decision_tree,
    run_dice,
    run_lime,
    train_rf_model,
)


def main() -> None:
    # 2) Load data
    cirrhosis = pd.read_csv(DATA_PATH)
    print(cirrhosis.shape)

    # 3) EDA
    run_eda(cirrhosis)

    # 4) Preprocessing

    # 4.1 Log transform
    cirrhosis = apply_log_transform(cirrhosis)

    # 4.2 Encode
    X, y = encode_features(cirrhosis)

    # 4.3 Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 4.4 Imputation
    X_train, X_test, imputation_results = select_best_imputation(
        X_train, X_test, y_train, y_test
    )

    # 4.5 Sampling
    X_train, y_train, sampling_results = select_best_sampling(
        X_train, y_train, X_test, y_test
    )

    # 4.6 Multiclass strategies
    multiclass_results = evaluate_multiclass_strategies(
        X_train, y_train, X_test, y_test
    )
    print("\nMulticlass strategy results:")
    print(multiclass_results)

    # 5) Explainability

    le, y_train_enc, y_test_enc = encode_labels(y_train, y_test)
    class_names = le.classes_

    model = train_rf_model(X_train, y_train_enc)
    print(classification_report(y_test_enc, model.predict(X_test), target_names=class_names))

    # 5.1 SHAP global
    explainer, shap_values = plot_shap_summary(model, X_test, class_names)

    # 5.2 SHAP waterfall
    plot_shap_waterfall(model, X_test, explainer, shap_values, class_names)

    # 5.3 PDP
    plot_pdp(model, X_train, class_names)

    # 5.4 ALE
    plot_ale(X_train, model)

    # 5.5 LIME
    run_lime(model, X_train, X_test)

    # 5.6 DiCE counterfactuals
    run_dice(model, X_train, X_test, y_train_enc)

    # 5.7 Intrinsic — Decision Tree
    run_decision_tree(X_train, y_train_enc, X_test, y_test_enc, class_names)


if __name__ == "__main__":
    main()