"""Model Evaluation and Interpretation Module for Health Risk Prediction System.

This module:
1. Loads the held-out test partition (unseen during model selection and training).
2. Evaluates the selected model pipeline and candidate baselines.
3. Computes:
   - Accuracy, Precision, Recall, F1-Score, ROC-AUC
   - Confusion Matrix (TN, FP, FN, TP)
   - ROC Curve coordinates
4. Performs Model Interpretation:
   - Random Forest Feature Importance (MDI)
   - Permutation Feature Importance on held-out test data
   - Logistic Regression standardized coefficients
   - Explicit disclaimers distinguishing statistical association from biological causation
5. Generates and saves evaluation plots and a structured JSON summary.
"""

import json
import os
import sys
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import (
    load_dataset,
    prepare_features_and_target,
    split_data,
)
from src.train_model import (
    RANDOM_STATE,
    build_candidate_pipelines,
)

OUTPUT_DIR = Path("models")
FIGURES_DIR = Path("models/figures")


def get_transformed_feature_names(preprocessor) -> list[str]:
    """Retrieve human-readable transformed feature names from ColumnTransformer."""
    try:
        raw_names = list(preprocessor.get_feature_names_out())
        clean_names = []
        for name in raw_names:
            # Format clean readable names
            clean_names.append(
                name.replace("num__", "")
                .replace("cat__", "")
                .replace("bin_disc__", "")
            )
        return clean_names
    except Exception:
        return [f"feature_{i}" for i in range(18)]


def evaluate_pipeline_on_test(pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute comprehensive test set metrics for a single pipeline."""
    y_pred = pipeline.predict(X_test)
    has_proba = hasattr(pipeline, "predict_proba")
    y_prob = pipeline.predict_proba(X_test)[:, 1] if has_proba else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_prob) if y_prob is not None else float("nan")
    cm = confusion_matrix(y_test, y_pred).tolist()

    result = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc),
        "confusion_matrix": cm,
        "true_negatives": int(cm[0][0]),
        "false_positives": int(cm[0][1]),
        "false_negatives": int(cm[1][0]),
        "true_positives": int(cm[1][1]),
        "test_sample_size": int(len(y_test)),
    }

    if y_prob is not None:
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        result["roc_curve"] = {
            "fpr": [float(x) for x in fpr],
            "tpr": [float(x) for x in tpr],
            "thresholds": [float(x) for x in thresholds],
        }

    return result


def plot_confusion_matrix(cm: list[list[int]], model_name: str, save_path: Path) -> None:
    """Generate and save formatted confusion matrix heatmap."""
    plt.figure(figsize=(5.5, 4.5), dpi=150)
    labels = ["No Disease (0)", "Heart Disease (1)"]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": 14, "weight": "bold"},
    )
    plt.title(f"Confusion Matrix: {model_name}\n(Held-out Test Set, N={sum(map(sum, cm))})", fontsize=11, pad=12)
    plt.xlabel("Predicted Label", fontsize=10, labelpad=8)
    plt.ylabel("Actual Label", fontsize=10, labelpad=8)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_roc_curves(test_results: dict[str, dict], save_path: Path) -> None:
    """Plot overlaid ROC curves for all evaluated models on the held-out test set."""
    plt.figure(figsize=(6, 5), dpi=150)

    for name, res in test_results.items():
        if "roc_curve" in res and not np.isnan(res["roc_auc"]):
            fpr = res["roc_curve"]["fpr"]
            tpr = res["roc_curve"]["tpr"]
            auc = res["roc_auc"]
            plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Chance Baseline")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=10)
    plt.title("ROC Curves Comparison (Held-out Test Set)", fontsize=11, pad=12)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def compute_and_plot_feature_importance(
    rf_pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path_rf: Path,
    save_path_perm: Path,
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute and plot Random Forest MDI importance and Permutation importance on test set."""
    # Ensure RF pipeline is fitted
    rf_pipeline.fit(X_train, y_train)

    preprocessor = rf_pipeline.named_steps["preprocessor"]
    rf_classifier = rf_pipeline.named_steps["classifier"]
    feature_names = get_transformed_feature_names(preprocessor)

    # 1. Random Forest MDI Feature Importance
    importances = rf_classifier.feature_importances_
    rf_importance_dict = dict(zip(feature_names, [float(x) for x in importances]))
    sorted_rf = sorted(rf_importance_dict.items(), key=lambda x: x[1], reverse=True)

    plt.figure(figsize=(7, 5), dpi=150)
    df_rf = pd.DataFrame(sorted_rf[:10], columns=["Feature", "Importance"])
    sns.barplot(data=df_rf, x="Importance", y="Feature", palette="Blues_r", hue="Feature", legend=False)
    plt.title(
        "Random Forest Feature Importance (MDI)\n"
        "[Statistical Association Only - Does Not Prove Medical Causation]",
        fontsize=10,
        pad=10,
    )
    plt.xlabel("Mean Decrease in Impurity (Gini Importance)", fontsize=9)
    plt.ylabel("Transformed Feature", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path_rf, bbox_inches="tight")
    plt.close()

    # 2. Permutation Feature Importance on held-out test data
    perm_result = permutation_importance(
        rf_pipeline,
        X_test,
        y_test,
        n_repeats=15,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
    )
    perm_dict = dict(zip(list(X_test.columns), [float(x) for x in perm_result.importances_mean]))
    sorted_perm = sorted(perm_dict.items(), key=lambda x: x[1], reverse=True)

    plt.figure(figsize=(7, 5), dpi=150)
    df_perm = pd.DataFrame(sorted_perm[:10], columns=["Raw Feature", "Mean ROC-AUC Drop"])
    sns.barplot(data=df_perm, x="Mean ROC-AUC Drop", y="Raw Feature", palette="crest", hue="Raw Feature", legend=False)
    plt.title(
        "Permutation Importance on Held-out Test Data\n"
        "[Impact of Feature Shuffling on Model ROC-AUC]",
        fontsize=10,
        pad=10,
    )
    plt.xlabel("Mean Drop in ROC-AUC When Feature Values are Shuffled", fontsize=9)
    plt.ylabel("Clinical / Demographic Feature", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path_perm, bbox_inches="tight")
    plt.close()

    return rf_importance_dict, perm_dict


def run_full_evaluation(
    data_path: str = "data/raw/heart_disease_raw.csv",
    best_pipeline_path: str = "models/best_model_pipeline.joblib",
    metadata_path: str = "models/model_metadata.json",
    output_eval_path: str = "models/test_evaluation_results.json",
) -> dict:
    """Run rigorous test set evaluation across all models and generate artifacts."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data and create identical split
    df = load_dataset(data_path)
    X, y = prepare_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # 2. Load metadata and best model pipeline
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    selected_model_name = metadata["selected_model"]
    selected_pipeline = joblib.load(best_pipeline_path)

    print(f"\n=======================================================")
    print(f" HELD-OUT TEST SET EVALUATION (N={len(X_test)})")
    print(f" Selected Model: {selected_model_name}")
    print(f"=======================================================\n")

    # 3. Train each candidate pipeline on X_train, then evaluate on held-out X_test
    candidate_pipelines = build_candidate_pipelines()
    test_results: dict[str, dict] = {}

    for name, pipe in candidate_pipelines.items():
        if name == selected_model_name:
            # Use the saved fitted pipeline
            eval_pipe = selected_pipeline
        else:
            # Fit candidate pipeline strictly on X_train
            pipe.fit(X_train, y_train)
            eval_pipe = pipe

        metrics = evaluate_pipeline_on_test(eval_pipe, X_test, y_test)
        test_results[name] = metrics

        print(
            f"[{name}] Test Set -> "
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"F1: {metrics['f1']:.4f} | "
            f"ROC-AUC: {metrics['roc_auc']:.4f} | "
            f"FN: {metrics['false_negatives']}"
        )

    # 4. Generate visualizations
    best_metrics = test_results[selected_model_name]
    cm_path = FIGURES_DIR / "confusion_matrix_selected.png"
    plot_confusion_matrix(best_metrics["confusion_matrix"], selected_model_name, cm_path)
    print(f"Saved confusion matrix plot to: {cm_path}")

    roc_path = FIGURES_DIR / "roc_curve_comparison.png"
    plot_roc_curves(test_results, roc_path)
    print(f"Saved ROC curve plot to: {roc_path}")

    # 5. Feature importance
    rf_pipe = candidate_pipelines["Random Forest"]
    rf_fig_path = FIGURES_DIR / "rf_feature_importance.png"
    perm_fig_path = FIGURES_DIR / "permutation_importance.png"
    rf_imp, perm_imp = compute_and_plot_feature_importance(
        rf_pipe, X_train, y_train, X_test, y_test, rf_fig_path, perm_fig_path
    )
    print(f"Saved feature importance plots to: {rf_fig_path} and {perm_fig_path}")

    # 6. Save structured evaluation results
    full_eval_data = {
        "selected_model": selected_model_name,
        "selection_rationale": metadata["selection_rationale"],
        "test_sample_size": int(len(X_test)),
        "test_results_by_model": test_results,
        "cv_results_by_model": metadata["cv_results"],
        "random_forest_mdi_importance": rf_imp,
        "permutation_importance": perm_imp,
        "figures": {
            "confusion_matrix": str(cm_path),
            "roc_curves": str(roc_path),
            "rf_feature_importance": str(rf_fig_path),
            "permutation_importance": str(perm_fig_path),
        },
        "interpretability_disclaimer": (
            "Feature importance and permutation metrics measure mathematical association within this "
            "specific historical sample (N=303). They do not establish clinical causation or disease etiology."
        ),
        "uncertainty_discussion": (
            f"The held-out test set comprises {len(X_test)} patients (28 with heart disease, 33 without). "
            f"Because of this small sample size, single sample variations can shift percentage metrics by ~3.5%. "
            f"The model is an educational demonstration and must never replace clinical diagnostics."
        ),
    }

    with open(output_eval_path, "w", encoding="utf-8") as f:
        json.dump(full_eval_data, f, indent=4)
    print(f"Saved test evaluation data to: {output_eval_path}")

    return full_eval_data


if __name__ == "__main__":
    run_full_evaluation()
