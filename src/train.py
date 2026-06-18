"""Train and benchmark the credit-risk classifiers, then persist artifacts.

Pipeline: clean data -> feature selection -> encode -> scale -> split ->
benchmark Random Forest vs XGBoost (tuned) -> save model + metrics + figures.
"""
from __future__ import annotations

import json

import joblib
import matplotlib
matplotlib.use("Agg")  # headless backend so figures save without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from . import config
from .data_prep import get_clean_data
from .feature_selection import select_features


def build_feature_matrix(df: pd.DataFrame, features: dict) -> pd.DataFrame:
    """Encode (ordinal EDUCATION + one-hot the rest) into a numeric design matrix."""
    cols = features["all"]
    work = df[cols + [config.TARGET]].copy()

    # Ordinal encoding for EDUCATION (qualification level).
    if "EDUCATION" in work.columns:
        work["EDUCATION"] = work["EDUCATION"].map(config.EDUCATION_MAP).astype(int)

    # One-hot encode the remaining nominal categoricals.
    onehot_cols = [c for c in features["categorical"] if c != "EDUCATION"]
    encoded = pd.get_dummies(work, columns=onehot_cols)
    return encoded


def scale_features(df_encoded: pd.DataFrame) -> pd.DataFrame:
    """Standard-scale the wide-range numeric columns in place."""
    out = df_encoded.copy()
    for col in config.COLUMNS_TO_SCALE:
        if col in out.columns:
            out[col] = StandardScaler().fit_transform(out[[col]])
    return out


def _save_confusion_matrix(y_true, y_pred, labels, path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted tier")
    plt.ylabel("Actual tier")
    plt.title("XGBoost confusion matrix (P1-P4)")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def _save_feature_importance(model, feature_names, path, top_n=20):
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in order]
    vals = importances[order]
    plt.figure(figsize=(8, 7))
    sns.barplot(x=vals, y=names, color="#2c7fb8")
    plt.xlabel("XGBoost feature importance")
    plt.title(f"Top {top_n} default-risk drivers")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return list(zip(names, [float(v) for v in vals]))


def train(verbose: bool = True) -> dict:
    """Run the full training pipeline and write all artifacts to disk."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ---- data + features --------------------------------------------------- #
    df = get_clean_data(verbose=verbose)
    features = select_features(df, verbose=verbose)
    encoded = build_feature_matrix(df, features)
    encoded = scale_features(encoded)

    y_labels = encoded[config.TARGET]
    X = encoded.drop(columns=[config.TARGET])
    feature_names = X.columns.tolist()

    # Encode P1-P4 labels to 0-3 for XGBoost.
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    metrics: dict = {"n_borrowers": int(df.shape[0]),
                     "n_model_features": len(feature_names)}

    # ---- Random Forest benchmark ------------------------------------------ #
    rf = RandomForestClassifier(**config.RF_PARAMS)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    metrics["random_forest_accuracy"] = round(float(rf_acc), 4)
    if verbose:
        print(f"\nRandom Forest accuracy : {rf_acc:.4f}")

    # ---- XGBoost (tuned best params) -------------------------------------- #
    xgb_clf = xgb.XGBClassifier(
        **config.XGB_BEST_PARAMS, random_state=config.RANDOM_STATE
    )
    xgb_clf.fit(X_train, y_train)
    xgb_pred = xgb_clf.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    metrics["xgboost_accuracy"] = round(float(xgb_acc), 4)
    metrics["best_model"] = "XGBoost"
    metrics["xgb_params"] = config.XGB_BEST_PARAMS
    if verbose:
        print(f"XGBoost accuracy (tuned): {xgb_acc:.4f}")

    # ---- reports & figures ------------------------------------------------- #
    tier_labels = list(label_encoder.classes_)
    y_test_tiers = label_encoder.inverse_transform(y_test)
    xgb_pred_tiers = label_encoder.inverse_transform(xgb_pred)

    report = classification_report(y_test_tiers, xgb_pred_tiers, digits=3)
    config.CLASSIFICATION_REPORT_PATH.write_text(
        "XGBoost classification report (test set)\n"
        "=========================================\n\n" + report, encoding="utf-8"
    )

    _save_confusion_matrix(
        y_test_tiers, xgb_pred_tiers, config.TIER_ORDER,
        config.FIGURES_DIR / "confusion_matrix.png",
    )
    top_drivers = _save_feature_importance(
        xgb_clf, feature_names, config.FIGURES_DIR / "feature_importance.png",
    )
    metrics["top_drivers"] = top_drivers[:10]

    with open(config.METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    # ---- persist model bundle --------------------------------------------- #
    bundle = {
        "model": xgb_clf,
        "feature_names": feature_names,
        "label_encoder": label_encoder,
        "tier_order": config.TIER_ORDER,
    }
    joblib.dump(bundle, config.MODEL_PATH)

    if verbose:
        print(f"\nSaved model   -> {config.MODEL_PATH}")
        print(f"Saved metrics -> {config.METRICS_PATH}")
        print(f"Saved figures -> {config.FIGURES_DIR}")
        print("\nTop default-risk drivers:")
        for name, val in top_drivers[:10]:
            print(f"  {name:<28} {val:.4f}")

    return metrics


if __name__ == "__main__":
    train()
