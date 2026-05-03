import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import shap
import dice_ml
from lime.lime_tabular import LimeTabularExplainer
from PyALE import ale
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

from config import RANDOM_STATE, TOP_FEATURES

# Model training helper
def train_rf_model(X_train: pd.DataFrame, y_train_enc: np.ndarray) -> RandomForestClassifier:
    """Train and return a Random Forest classifier."""
    model = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
    model.fit(X_train, y_train_enc)
    return model


def encode_labels(y_train: pd.Series, y_test: pd.Series):
    """Encode string labels to integers. Returns (le, y_train_enc, y_test_enc)."""
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    return le, y_train_enc, y_test_enc

# 5.1  SHAP — global summary
def plot_shap_summary(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    class_names,
) -> tuple:
    """Compute SHAP values and plot global bar + per-class beeswarm."""
    explainer = shap.TreeExplainer(model)
    sv_raw = explainer.shap_values(X_test)
    shap_values = [sv_raw[:, :, i] for i in range(len(class_names))]

    shap.summary_plot(
        shap_values, X_test,
        feature_names=X_test.columns.tolist(),
        class_names=class_names,
        plot_type="bar",
    )

    for i, cls in enumerate(class_names):
        print(f"Class: {cls}")
        shap.summary_plot(shap_values[i], X_test, feature_names=X_test.columns.tolist())

    return explainer, shap_values


# 5.2  SHAP — local waterfall
def plot_shap_waterfall(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    explainer,
    shap_values: list,
    class_names,
) -> None:
    """Waterfall plot for the most confident prediction per class."""
    proba_test = model.predict_proba(X_test)
    rep_idx = [np.argmax(proba_test[:, i]) for i in range(len(class_names))]

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle(
        "SHAP Waterfall — Most Confident Prediction per Class",
        fontsize=15, fontweight="bold",
    )

    for i, (cls, idx) in enumerate(zip(class_names, rep_idx)):
        plt.sca(axes[i])
        sv = shap.Explanation(
            values=shap_values[i][idx],
            base_values=explainer.expected_value[i],
            data=X_test.iloc[idx].values,
            feature_names=X_test.columns.tolist(),
        )
        shap.waterfall_plot(sv, max_display=12, show=False)
        axes[i].set_title(
            f"Class: {cls}  (p = {proba_test[idx, i]:.2f})",
            fontsize=12, fontweight="bold",
        )

    plt.tight_layout()
    plt.show()

# 5.3  PDP (Partial Dependence Plots)
def plot_pdp(
    model: RandomForestClassifier,
    X_train: pd.DataFrame,
    class_names,
    features: list = None,
) -> None:
    """Plot PDP for each class and each top feature."""
    if features is None:
        features = TOP_FEATURES

    for i, cls in enumerate(class_names):
        fig, ax = plt.subplots(1, len(features), figsize=(18, 4))
        fig.suptitle(f"Partial Dependence Plots — Class: {cls}", fontsize=13, fontweight="bold")
        PartialDependenceDisplay.from_estimator(
            model, X_train, features=features, target=i, ax=ax
        )
        plt.tight_layout()
        plt.show()

# 5.4  ALE (Accumulated Local Effects)
def plot_ale(X_train: pd.DataFrame, model, features: list = None) -> None:
    """Plot ALE for each top feature."""
    if features is None:
        features = TOP_FEATURES

    for feat in features:
        ale_eff = ale(X=X_train, model=model, feature=[feat], grid_size=50, plot=True)
        plt.title(f"ALE — {feat}")
        plt.show()

# 5.5  LIME
def plot_lime_explanation(
    exp,
    class_idx: int,
    class_name: str,
    predict_proba,
    instance: np.ndarray,
    model_classes,
    num_features: int = 10,
) -> None:
    """
    Plots a LIME explanation for a given instance and class.

    Parameters
    ----------
    exp          : LIME explanation object
    class_idx    : index of the class being explained
    class_name   : name of the class being explained
    predict_proba: function to get prediction probabilities from the model
    instance     : the data instance being explained (1-D array)
    model_classes: array of class labels
    num_features : number of top features to display
    """
    exp_list = sorted(exp.as_list(label=class_idx), key=lambda x: x[1])
    feats = [e[0] for e in exp_list]
    weights = [e[1] for e in exp_list]
    colors = ["#2ecc71" if w > 0 else "#e74c3c" for w in weights]

    probas = predict_proba([instance])[0]

    fig, (ax_bar, ax_prob) = plt.subplots(
        1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [3, 1]}
    )
    fig.suptitle(f"LIME Explanation — Class: {class_name}", fontsize=14, fontweight="bold")

    bars = ax_bar.barh(feats, weights, color=colors, edgecolor="white", linewidth=0.5)
    ax_bar.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    ax_bar.set_xlabel("Feature weight", fontsize=11)
    ax_bar.set_title("Feature contributions", fontsize=12)
    ax_bar.spines[["top", "right"]].set_visible(False)
    for bar, w in zip(bars, weights):
        ax_bar.text(
            w + 0.001 if w >= 0 else w - 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.3f}", va="center",
            ha="left" if w >= 0 else "right", fontsize=8,
        )

    bar_colors = ["#2ecc71" if i == class_idx else "#aab4be" for i in range(len(probas))]
    ax_prob.barh(model_classes, probas, color=bar_colors, edgecolor="white")
    ax_prob.set_xlim(0, 1)
    ax_prob.set_xlabel("Probability", fontsize=11)
    ax_prob.set_title("Prediction probabilities", fontsize=12)
    ax_prob.spines[["top", "right"]].set_visible(False)
    for i, p in enumerate(probas):
        ax_prob.text(p + 0.01, i, f"{p:.2f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.show()


def run_lime(
    model: RandomForestClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> None:
    """Run LIME for the most confident prediction of each class."""
    lime_exp = LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=model.classes_,
        mode="classification",
        random_state=RANDOM_STATE,
    )

    proba_test = model.predict_proba(X_test)

    for i, cls in enumerate(model.classes_):
        idx = np.argmax(proba_test[:, i])
        exp = lime_exp.explain_instance(
            X_test.iloc[idx].values,
            model.predict_proba,
            num_features=10,
            labels=[i],
        )
        plot_lime_explanation(exp, i, cls, model.predict_proba,
                              X_test.iloc[idx].values, model.classes_)

# 5.6  DiCE — Counterfactual Explanations
def run_dice(
    model: RandomForestClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_enc: np.ndarray,
) -> None:
    """Generate and display DiCE counterfactuals for the most confident deceased patient."""
    train_df = X_train.copy()
    train_df["Status"] = y_train_enc

    d = dice_ml.Data(
        dataframe=train_df,
        continuous_features=X_train.columns.tolist(),
        outcome_name="Status",
    )
    m = dice_ml.Model(model=model, backend="sklearn")
    exp = dice_ml.Dice(d, m, method="random")

    proba_test = model.predict_proba(X_test)
    idx_deceased = np.argmax(proba_test[:, 2])
    query_instance = X_test.iloc[[idx_deceased]]

    cf = exp.generate_counterfactuals(
        query_instance,
        total_CFs=3,
        desired_class=0,
        permitted_range={
            "log_Bilirubin": [X_train["log_Bilirubin"].min(), X_train["log_Bilirubin"].max()],
            "Stage": [1, 4],
            "Prothrombin": [X_train["Prothrombin"].min(), X_train["Prothrombin"].max()],
        },
    )
    cf.visualize_as_dataframe(show_only_changes=True)

# 5.7  Intrinsic explainability — Decision Tree
def run_decision_tree(
    X_train: pd.DataFrame,
    y_train_enc: np.ndarray,
    X_test: pd.DataFrame,
    y_test_enc: np.ndarray,
    class_names,
) -> DecisionTreeClassifier:
    """Train a shallow Decision Tree, print report, plot the tree and feature importances."""
    dt = DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=RANDOM_STATE)
    dt.fit(X_train, y_train_enc)

    print(classification_report(y_test_enc, dt.predict(X_test), target_names=class_names))

    fig, ax = plt.subplots(figsize=(20, 8))
    plot_tree(
        dt, feature_names=X_train.columns.tolist(),
        class_names=class_names, filled=True, fontsize=8, ax=ax,
    )
    plt.title("Decision Tree: Intrinsic Explainability", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()

    # Feature importances
    importances = dt.feature_importances_
    sorted_idx = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        [X_train.columns[i] for i in sorted_idx if importances[i] > 0],
        [importances[i] for i in sorted_idx if importances[i] > 0],
        color="#4c8bb5", edgecolor="white",
    )
    ax.set_xlabel("Feature importance (Gini)", fontsize=11)
    ax.set_title("Decision Tree: Feature Importance", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.show()

    return dt