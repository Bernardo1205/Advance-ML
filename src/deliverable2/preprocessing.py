import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import LinearSVC
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from missforest import MissForest

from config import RANDOM_STATE, SKEWED_COLS, TARGET_COLUMN



# 4.1  Log transformation
def apply_log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Drop ID, convert Age to years, apply log1p to skewed columns."""
    df = df.drop(columns=["ID"])
    df["Age"] = (df["Age"] / 365.25).round(2)

    for col in SKEWED_COLS:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col])
    df = df.drop(columns=[c for c in SKEWED_COLS if c in df.columns])
    return df

# 4.2  Encode categorical variables
def encode_features(df: pd.DataFrame):
    """
    Split into X / y and one-hot encode categorical columns.

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    cat_cols = X.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=int)
    return X, y

# 4.3  Train/test split
def split_data(X: pd.DataFrame, y: pd.Series):
    """Stratified 80/20 split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    return X_train, X_test, y_train, y_test

# 4.4  Imputation strategies
def mice_imputation(train: pd.DataFrame, test: pd.DataFrame):
    imputer = IterativeImputer(random_state=RANDOM_STATE)
    train_imp = pd.DataFrame(imputer.fit_transform(train), columns=train.columns)
    test_imp = pd.DataFrame(imputer.transform(test), columns=test.columns)
    return train_imp, test_imp


def forest_imputation(train: pd.DataFrame, test: pd.DataFrame):
    imputer = MissForest()
    train_imp = pd.DataFrame(imputer.fit_transform(train), columns=train.columns)
    test_imp = pd.DataFrame(imputer.transform(test), columns=test.columns)
    return train_imp, test_imp


def knn_imputation(train: pd.DataFrame, test: pd.DataFrame):
    imputer = KNNImputer()
    train_imp = pd.DataFrame(imputer.fit_transform(train), columns=train.columns)
    test_imp = pd.DataFrame(imputer.transform(test), columns=test.columns)
    return train_imp, test_imp


IMPUTATION_STRATEGIES = {
    "MICE": mice_imputation,
    "MissForest": forest_imputation,
    "KNN": knn_imputation,
}


def evaluate_imputation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    strategies: dict,
) -> pd.DataFrame:
    """Train a RF for each imputation strategy and return macro F1 scores."""
    results = []
    for name, imputer_fn in strategies.items():
        X_tr_imp, X_te_imp = imputer_fn(X_train, X_test)
        model = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
        model.fit(X_tr_imp, y_train)
        y_pred = model.predict(X_te_imp)
        results.append({
            "strategy": name,
            "f1_score": f1_score(y_test, y_pred, average="macro"),
        })
    return pd.DataFrame(results)


def select_best_imputation(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
):
    """Evaluate all imputation strategies and apply the best one."""
    results_df = evaluate_imputation(
        X_train, y_train, X_test, y_test, IMPUTATION_STRATEGIES
    )
    print(results_df)

    best = results_df.sort_values("f1_score", ascending=False)["strategy"].iloc[0]
    print(f"\nBest imputation strategy: {best}")

    impute_fn = IMPUTATION_STRATEGIES[best]
    X_train_imp, X_test_imp = impute_fn(X_train, X_test)
    return X_train_imp, X_test_imp, results_df

# 4.5  Sampling strategies
SAMPLING_STRATEGIES = {
    "baseline": None,
    "ENN (Under-sampling)": EditedNearestNeighbours(),
    "SMOTE (Over-sampling)": SMOTE(random_state=RANDOM_STATE),
    "SMOTEENN (Combined)": SMOTEENN(random_state=RANDOM_STATE),
}


def evaluate_sampling(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    strategies: dict,
) -> pd.DataFrame:
    """Train a RF for each sampling strategy and return accuracy scores."""
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
        })
    return pd.DataFrame(results)


def select_best_sampling(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    """Evaluate all sampling strategies and apply the best one."""
    results_df = evaluate_sampling(
        X_train, y_train, X_test, y_test, SAMPLING_STRATEGIES
    )
    print(results_df)

    best = results_df.sort_values("accuracy", ascending=False)["strategy"].iloc[0]
    print(f"\nBest sampling strategy: {best}")

    sampler = SAMPLING_STRATEGIES[best]
    if sampler is not None:
        X_train, y_train = sampler.fit_resample(X_train, y_train)
    return X_train, y_train, results_df


# 4.6  Multiclass strategy comparison
MULTICLASS_MODELS = {
    "Softmax": LogisticRegression(
        max_iter=3000,
        random_state=RANDOM_STATE,
    ),
    "One-vs-Rest": OneVsRestClassifier(LinearSVC(random_state=RANDOM_STATE)),
    "One-vs-One": OneVsOneClassifier(LinearSVC(random_state=RANDOM_STATE)),
}


def evaluate_multiclass_strategies(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models: dict = None,
) -> pd.DataFrame:
    """Fit each multiclass model and return accuracy scores."""
    if models is None:
        models = MULTICLASS_MODELS
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rows.append({
            "strategy": name,
            "accuracy": accuracy_score(y_test, y_pred),
        })
    return pd.DataFrame(rows)