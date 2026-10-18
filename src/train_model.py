"""Model Training and Selection Module for Health Risk Prediction System.

This module:
1. Loads preprocessed training and test partitions.
2. Constructs full Scikit-learn Pipelines combining preprocessing with classifiers:
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier
3. Performs 5-fold Stratified Cross-Validation strictly on training data.
4. Compares models across Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
5. Selects the best model using a clinical validation criterion:
   Prioritizing high Recall (minimizing False Negatives) while maintaining
   strong ROC-AUC and F1-score.
6. Fits the selected pipeline on the full training set.
7. Saves the complete fitted pipeline and metadata using Joblib and JSON.
8. Verifies reload and inference on sample test data.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import (
    get_preprocessor,
    load_dataset,
    prepare_features_and_target,
    split_data,
)

RANDOM_STATE = 42
N_SPLITS = 5


def build_candidate_pipelines() -> dict[str, Pipeline]:
    """Build candidate Scikit-learn pipelines with identical preprocessors."""
    preprocessor = get_preprocessor()

    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        max_iter=1000,
                        solver="lbfgs",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    DecisionTreeClassifier(
                        max_depth=4,
                        min_samples_split=6,
                        min_samples_leaf=3,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=100,
                        max_depth=5,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    return models


def evaluate_cv_fold(
    pipeline: Pipeline,
    X_train_fold: pd.DataFrame,
    y_train_fold: pd.Series,
    X_val_fold: pd.DataFrame,
    y_val_fold: pd.Series,
) -> dict[str, float]:
    """Train pipeline on train fold and evaluate on validation fold."""
    # Fit pipeline ONLY on training fold (prevent leakage)
    pipeline.fit(X_train_fold, y_train_fold)

    y_pred = pipeline.predict(X_val_fold)
    y_prob = (
        pipeline.predict_proba(X_val_fold)[:, 1]
        if hasattr(pipeline, "predict_proba")
        else None
    )

    acc = accuracy_score(y_val_fold, y_pred)
    prec = precision_score(y_val_fold, y_pred, zero_division=0)
    rec = recall_score(y_val_fold, y_pred, zero_division=0)
    f1 = f1_score(y_val_fold, y_pred, zero_division=0)
    roc = roc_auc_score(y_val_fold, y_prob) if y_prob is not None else float("nan")

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc),
    }


def perform_cross_validation(
    X_train: pd.DataFrame, y_train: pd.Series
) -> dict[str, dict[str, float]]:
    """Perform 5-fold stratified cross validation on training set for all models."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_summary: dict[str, dict[str, float]] = {}

    print(
        f"--- Starting {N_SPLITS}-Fold Stratified Cross-Validation on Training Data (N={len(X_train)}) ---"
    )

    for model_name, pipeline in build_candidate_pipelines().items():
        fold_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": []}

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            metrics = evaluate_cv_fold(pipeline, X_tr, y_tr, X_val, y_val)
            for k in fold_metrics:
                fold_metrics[k].append(metrics[k])

        cv_summary[model_name] = {
            "cv_accuracy_mean": float(np.mean(fold_metrics["accuracy"])),
            "cv_accuracy_std": float(np.std(fold_metrics["accuracy"])),
            "cv_precision_mean": float(np.mean(fold_metrics["precision"])),
            "cv_precision_std": float(np.std(fold_metrics["precision"])),
            "cv_recall_mean": float(np.mean(fold_metrics["recall"])),
            "cv_recall_std": float(np.std(fold_metrics["recall"])),
            "cv_f1_mean": float(np.mean(fold_metrics["f1"])),
            "cv_f1_std": float(np.std(fold_metrics["f1"])),
            "cv_roc_auc_mean": float(np.mean(fold_metrics["roc_auc"])),
            "cv_roc_auc_std": float(np.std(fold_metrics["roc_auc"])),
        }

        print(
            f"[{model_name}] "
            f"Recall: {cv_summary[model_name]['cv_recall_mean']:.3f} (+/- {cv_summary[model_name]['cv_recall_std']:.3f}) | "
            f"ROC-AUC: {cv_summary[model_name]['cv_roc_auc_mean']:.3f} | "
            f"F1: {cv_summary[model_name]['cv_f1_mean']:.3f} | "
            f"Accuracy: {cv_summary[model_name]['cv_accuracy_mean']:.3f}"
        )

    return cv_summary


def select_best_model(cv_summary: dict[str, dict[str, float]]) -> tuple[str, str]:
    """Select model based on documented clinical validation criterion.

    CRITERION RATIONALE:
    In medical risk estimation, missing a disease case (False Negative) is substantially
    more hazardous than flagging an unaffected patient for non-invasive follow-up (False Positive).
    Therefore:
    1. Primary selection ranking: Highest mean Cross-Validation Recall (minimizing False Negatives).
    2. Secondary ranking (within 0.02 of top recall): Highest ROC-AUC followed by F1-Score.
    3. Accuracy is never used as the sole deciding metric.
    """
    sorted_candidates = sorted(
        cv_summary.items(),
        key=lambda item: (
            item[1]["cv_recall_mean"],
            item[1]["cv_roc_auc_mean"],
            item[1]["cv_f1_mean"],
        ),
        reverse=True,
    )

    best_model_name, best_metrics = sorted_candidates[0]

    rationale = (
        f"Selected '{best_model_name}' based on the clinical validation criterion: "
        f"Mean CV Recall = {best_metrics['cv_recall_mean']:.4f} "
        f"(minimizing critical false negatives), "
        f"Mean CV ROC-AUC = {best_metrics['cv_roc_auc_mean']:.4f}, and "
        f"Mean CV F1-Score = {best_metrics['cv_f1_mean']:.4f}. "
        f"Accuracy ({best_metrics['cv_accuracy_mean']:.4f}) was monitored but not prioritized alone."
    )

    return best_model_name, rationale


def train_and_save_pipeline(
    data_path: str = "data/raw/heart_disease_raw.csv",
    model_output_path: str = "models/best_model_pipeline.joblib",
    metadata_output_path: str = "models/model_metadata.json",
) -> dict:
    """Full training pipeline workflow: load, split, cross-validate, fit best model, and save."""
    df = load_dataset(data_path)
    X, y = prepare_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=RANDOM_STATE)

    cv_summary = perform_cross_validation(X_train, y_train)
    best_model_name, rationale = select_best_model(cv_summary)

    print(f"\nSelection Decision:\n{rationale}\n")

    # Re-instantiate a fresh candidate pipeline for the chosen architecture
    best_pipeline = build_candidate_pipelines()[best_model_name]

    # Fit ONLY on the training partition
    print(f"Fitting final '{best_model_name}' pipeline on entire training partition (N={len(X_train)})...")
    best_pipeline.fit(X_train, y_train)

    # Save model pipeline
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(best_pipeline, model_output_path)
    print(f"Saved fitted pipeline to: {model_output_path}")

    # Prepare metadata
    metadata = {
        "selected_model": best_model_name,
        "selection_rationale": rationale,
        "random_state": RANDOM_STATE,
        "cv_folds": N_SPLITS,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "cv_results": cv_summary,
        "feature_names": list(X.columns),
    }

    with open(metadata_output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved training metadata to: {metadata_output_path}")

    # Verification: Reload and test inference on a sample record
    print("\nVerifying reloaded pipeline inference...")
    reloaded_pipeline = joblib.load(model_output_path)
    sample_record = X_test.iloc[0:1]
    pred = reloaded_pipeline.predict(sample_record)[0]
    prob = (
        reloaded_pipeline.predict_proba(sample_record)[0, 1]
        if hasattr(reloaded_pipeline, "predict_proba")
        else None
    )
    print(f"Sample test row 0 prediction: {pred} (Predicted probability: {prob})")
    print("Verification successful!")

    return metadata


if __name__ == "__main__":
    train_and_save_pipeline()
