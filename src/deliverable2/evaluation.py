"""Evaluate the deep learning model and save a clear confusion matrix.

This script reloads the saved checkpoint, recreates the preprocessing flow,
and prints classification metrics. It also saves a confusion matrix figure
with both raw counts and row-normalized percentages so the results are easier
to interpret.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DATA_PATH, RANDOM_STATE  # noqa: E402
from data import prepare_data  # noqa: E402
from model import CirrhosisNN  # noqa: E402
from preprocessing import (  # noqa: E402
    apply_log_transform,
    encode_features,
    select_best_imputation,
    select_best_sampling,
    split_data,
)
from train_utils import predict  # noqa: E402


def _resolve_path(path_str: str | Path) -> Path:
    """Resolve a path from several common locations in the project."""
    path = Path(path_str)
    if path.is_absolute() and path.exists():
        return path

    for candidate in (path, CURRENT_DIR / path, ROOT_DIR / path):
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate

    return (ROOT_DIR / path).resolve()


def _build_loaders(csv_path: Path):
    """Recreate the preprocessing pipeline and return train/val/test loaders."""
    cirrhosis = pd.read_csv(csv_path)
    cirrhosis = apply_log_transform(cirrhosis)

    features, target = encode_features(cirrhosis)
    features = cast(pd.DataFrame, cast(object, features))
    target = cast(pd.Series, cast(object, target))

    X_train, X_test, y_train, y_test = split_data(features, target)
    X_train, X_test, _ = select_best_imputation(X_train, X_test, y_train, y_test)
    X_train, y_train, _ = select_best_sampling(X_train, y_train, X_test, y_test)

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    train_loader, val_loader, test_loader, scaler_x, scaler_y = prepare_data(
        X_fit, X_test, X_val, y_fit, y_test, y_val
    )
    return train_loader, val_loader, test_loader, scaler_x, scaler_y


def _load_model(model_path: Path, device: torch.device) -> CirrhosisNN:
    """Load the trained neural network from the checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    best = checkpoint["best_params"]

    model = CirrhosisNN(best["h1"], best["h2"], best["h3"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _save_confusion_matrix(y_true, y_pred, class_names, output_path: Path) -> None:
    """Save a confusion matrix figure with counts and normalized percentages."""
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    cm_norm = confusion_matrix(y_true, y_pred, labels=class_names, normalize="true")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0],
    )
    axes[0].set_title("Confusion Matrix - Counts")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Real")

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Greens",
        vmin=0,
        vmax=1,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1],
    )
    axes[1].set_title("Confusion Matrix - Row Normalized")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Real")

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_evaluation(
    model_path: str = "src/deliverable2/model.pt",
    csv_path: str = DATA_PATH,
    output_dir: str = "reports",
    n_examples: int = 10,
) -> Path:
    """Evaluate the trained model and save the outputs to disk."""
    model_path = _resolve_path(model_path)
    csv_path = _resolve_path(csv_path)
    output_dir = _resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, _, test_loader, _, scaler_y = _build_loaders(csv_path)
    model = _load_model(model_path, device)

    y_pred, y_true = predict(model, test_loader, device, scaler_y)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    if hasattr(scaler_y, "classes_"):
        class_names = list(scaler_y.classes_)
    else:
        class_names = sorted(pd.unique(pd.Series(y_true)).tolist())

    report = classification_report(
        y_true,
        y_pred,
        labels=class_names,
        target_names=class_names,
        zero_division=0,
    )

    cm_path = "dl_confusion_matrix.png"
    _save_confusion_matrix(y_true, y_pred, class_names, cm_path)

    summary_path = output_dir / "dl_model_evaluation.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Deep Learning Model Evaluation\n")
        f.write("=" * 32 + "\n\n")
        f.write(f"Model path: {model_path}\n")
        f.write(f"Data path: {csv_path}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n")
        f.write(f"Macro Precision: {macro_precision:.4f}\n")
        f.write(f"Macro Recall: {macro_recall:.4f}\n\n")
        f.write("Classification report:\n")
        f.write(report)

    print("\n=== Evaluation Summary ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print("\nClassification report:\n")
    print(report)
    print(f"\nConfusion matrix saved to: {cm_path}")
    print(f"Text summary saved to: {summary_path}")

    print("\nFirst predictions:")
    for i in range(min(n_examples, len(y_true))):
        print(f"Real: {y_true[i]} | Pred: {y_pred[i]}")

    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the Deep Learning model for deliverable2.")
    parser.add_argument("--model-path", default="src/deliverable2/model.pt", help="Path to the saved model checkpoint.")
    parser.add_argument("--csv-path", default=DATA_PATH, help="Path to the cirrhosis dataset CSV.")
    parser.add_argument("--output-dir", default="reports", help="Folder where the evaluation outputs will be saved.")
    parser.add_argument("--n-examples", type=int, default=10, help="How many example predictions to print.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_evaluation(
        model_path=args.model_path,
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        n_examples=args.n_examples,
    )


if __name__ == "__main__":
    main()

