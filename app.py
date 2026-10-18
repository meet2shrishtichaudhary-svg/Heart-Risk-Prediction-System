"""Health Risk Prediction System - Streamlit Web Application.

A professional, educational Machine Learning application built with Streamlit,
Pandas, NumPy, and Scikit-learn using the UCI Heart Disease (Cleveland) dataset.
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
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import (
    COLUMN_NAMES,
    FEATURE_NAMES,
    download_raw_dataset,
    load_dataset,
    prepare_features_and_target,
    split_data,
)
from src.train_model import (
    RANDOM_STATE,
    train_and_save_pipeline,
)
from src.evaluate_model import run_full_evaluation

# Set page configuration
st.set_page_config(
    page_title="Health Risk Prediction System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern medical UI styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 24px 32px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 14px rgba(30, 58, 138, 0.15);
    }

    .main-header h1 {
        color: white !important;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .main-header p {
        color: #dbeafe !important;
        font-size: 1.05rem;
        margin-bottom: 0px;
    }

    .disclaimer-banner {
        background-color: #fef2f2;
        border-left: 5px solid #ef4444;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 24px;
        color: #991b1b;
        font-size: 0.95rem;
    }

    .info-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    .metric-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .badge-blue { background-color: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; }
    .badge-green { background-color: #f0fdf4; color: #15803d; border: 1px solid #bbf7d0; }
    .badge-amber { background-color: #fffbeb; color: #b45309; border: 1px solid #fde68a; }

    .result-box-positive {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 2px solid #ef4444;
        border-radius: 14px;
        padding: 24px;
        color: #991b1b;
        margin-top: 20px;
    }

    .result-box-negative {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 2px solid #22c55e;
        border-radius: 14px;
        padding: 24px;
        color: #166534;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_dataset():
    """Load and cache dataset."""
    raw_path = "data/raw/heart_disease_raw.csv"
    if not os.path.exists(raw_path):
        download_raw_dataset(raw_path)
    return load_dataset(raw_path)


@st.cache_resource
def get_trained_pipeline():
    """Load cached model pipeline or train if not present."""
    model_path = "models/best_model_pipeline.joblib"
    eval_path = "models/test_evaluation_results.json"
    if not os.path.exists(model_path) or not os.path.exists(eval_path):
        with st.spinner("Initializing models and running reproducible evaluation..."):
            train_and_save_pipeline()
            run_full_evaluation()
    pipeline = joblib.load(model_path)
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    return pipeline, eval_data


# Sidebar Navigation
st.sidebar.markdown("## 🏥 System Navigation")
selected_page = st.sidebar.radio(
    "Go to",
    ["Home", "Dataset Overview", "Model Evaluation", "Health Risk Prediction", "About"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Project Details**
    - **Dataset:** UCI Heart Disease (Cleveland)
    - **Algorithms:** Logistic Regression, Decision Tree, Random Forest
    - **Optimization Goal:** Recall & False Negative Minimization
    - **Environment:** Scikit-learn & Streamlit
    """
)

# Load data and pipeline
try:
    df_raw = get_dataset()
    pipeline, eval_data = get_trained_pipeline()
except Exception as e:
    st.error(f"Error initializing system assets: {e}")
    st.stop()


# ==============================================================================
# 1. HOME PAGE
# ==============================================================================
if selected_page == "Home":
    st.markdown(
        """
        <div class="main-header">
            <h1>Health Risk Prediction System</h1>
            <p>An Educational Machine Learning Framework for Preventive Heart Disease Risk Stratification</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="disclaimer-banner">
            ⚠️ <strong>Academic & Educational Disclaimer:</strong> This system is developed strictly for educational
            demonstration and academic research as part of a B.Tech CSE (AI & ML) Major Project. The predictions, probabilities,
            and feature associations produced are mathematical estimates and <strong>MUST NOT</strong> be used as a medical diagnosis
            or clinical decision support. Always consult a certified healthcare professional for medical concerns.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### 📌 Problem Statement & Objective")
        st.write(
            """
            Cardiovascular diseases represent the leading cause of mortality globally. Early detection of clinical warning
            signals allows timely lifestyle interventions and non-invasive secondary prevention.
            
            This project investigates how classical supervised machine learning classifiers can be rigorously trained,
            statistically evaluated, and responsibly interpreted on clinical patient records from the renowned
            **UCI Heart Disease (Cleveland)** dataset.
            """
        )

        st.markdown("### ⚙️ How the System Operates")
        st.markdown(
            """
            1. **Data Ingestion & Integrity:** Ingests 303 historical patient records comprising 13 demographic and non-invasive diagnostic indicators.
            2. **Leakage-Free Preprocessing:** Standardizes continuous vitals (StandardScaler) and encodes clinical categories (OneHotEncoder) strictly inside Scikit-learn pipelines fitted on training data.
            3. **Clinical Model Selection:** Compares Logistic Regression, Decision Tree, and Random Forest using 5-Fold Stratified Cross-Validation, actively selecting the model that optimizes **Recall** to minimize dangerous False Negatives.
            4. **Transparent Risk Stratification:** Provides interactive single-patient inference, uncalibrated probability scores, and feature importance visual explanations.
            """
        )

    with col2:
        st.markdown("### 📊 System At a Glance")
        st.markdown(
            f"""
            <div class="info-card">
                <h4>Dataset Provenance</h4>
                <p><strong>Source:</strong> UCI Machine Learning Repository</p>
                <p><strong>Cohort:</strong> Cleveland Clinic Foundation (Dr. Robert Detrano et al.)</p>
                <p><strong>Total Instances:</strong> {len(df_raw)} records (13 predictors + target)</p>
                <hr style="margin: 10px 0;">
                <h4>Selected Classification Model</h4>
                <p><strong>Model:</strong> <span class="metric-badge badge-blue">{eval_data['selected_model']}</span></p>
                <p><strong>Test Recall (Sensitivity):</strong> <span class="metric-badge badge-green">{eval_data['test_results_by_model'][eval_data['selected_model']]['recall'] * 100:.1f}%</span></p>
                <p><strong>Test ROC-AUC:</strong> <span class="metric-badge badge-blue">{eval_data['test_results_by_model'][eval_data['selected_model']]['roc_auc']:.3f}</span></p>
                <p><strong>False Negatives on Test Set:</strong> <span class="metric-badge badge-amber">{eval_data['test_results_by_model'][eval_data['selected_model']]['false_negatives']} of 28 positive cases</span></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.info("💡 **Quick Navigation:** Use the sidebar menu on the left to explore the Dataset, view Model Evaluation benchmarks, or test an individual patient in the Health Risk Prediction tab.")


# ==============================================================================
# 2. DATASET OVERVIEW
# ==============================================================================
elif selected_page == "Dataset Overview":
    st.markdown("## 🔍 Dataset Overview & Exploratory Summary")
    st.markdown(
        "Explore dataset dimensions, missing value audits, class balance, and clinical attribute distributions."
    )

    df_clean = df_raw.copy()
    df_clean["target"] = (df_clean["num"] > 0).astype(int)

    # Top summary metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Records", len(df_raw))
    m2.metric("Predictive Attributes", 13)
    m3.metric("Missing Value Records", "6 (1.98%)")
    m4.metric("Heart Disease Prevalence", f"{(df_clean['target'].mean() * 100):.1f}%")

    tab1, tab2, tab3 = st.tabs(["Data Table & Missing Audit", "Target Distribution", "Clinical Attributes Distribution"])

    with tab1:
        st.markdown("### Raw Patient Records (First 15 Rows)")
        st.dataframe(df_raw.head(15), use_container_width=True)

        st.markdown("### Missing Value Summary")
        null_counts = df_raw.isnull().sum()
        null_df = pd.DataFrame(
            {
                "Attribute": null_counts.index,
                "Missing Values": null_counts.values,
                "Percentage (%)": (null_counts.values / len(df_raw) * 100).round(2),
                "Handled Via": [
                    "Most Frequent Imputation" if count > 0 else "None (Complete)"
                    for count in null_counts.values
                ],
            }
        )
        st.dataframe(null_df, use_container_width=True)

    with tab2:
        st.markdown("### Target Class Distribution (Absence vs Presence)")
        col_t1, col_t2 = st.columns(2)

        with col_t1:
            fig_t1, ax_t1 = plt.subplots(figsize=(6, 4))
            counts = df_clean["target"].value_counts().rename({0: "No Disease (0)", 1: "Heart Disease (1)"})
            ax_t1.pie(
                counts,
                labels=counts.index,
                autopct="%1.1f%%",
                colors=["#3b82f6", "#ef4444"],
                startangle=140,
                explode=[0.02, 0.02],
            )
            ax_t1.set_title("Binarized Disease Status Proportion")
            st.pyplot(fig_t1)
            plt.close()

        with col_t2:
            fig_t2, ax_t2 = plt.subplots(figsize=(6, 4))
            sns.countplot(data=df_raw, x="num", ax=ax_t2, palette="Blues_d")
            ax_t2.set_title("Original Angiographic Disease Severity (num 0 to 4)")
            ax_t2.set_xlabel("Severity Grade (0: Absence; 1-4: Varying Narrowing)")
            ax_t2.set_ylabel("Patient Count")
            st.pyplot(fig_t2)
            plt.close()

    with tab3:
        st.markdown("### Exploratory Visualizations: Key Biomarkers vs Health Status")
        feature_choice = st.selectbox(
            "Select Continuous Biomarker to Inspect:",
            ["age", "trestbps", "chol", "thalach", "oldpeak"],
            format_func=lambda x: {
                "age": "Age (Years)",
                "trestbps": "Resting Blood Pressure (trestbps in mm Hg)",
                "chol": "Serum Cholestoral (chol in mg/dl)",
                "thalach": "Max Heart Rate Achieved (thalach in bpm)",
                "oldpeak": "ST Depression (oldpeak in mm)",
            }[x],
        )

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            fig_f1, ax_f1 = plt.subplots(figsize=(6, 4))
            sns.histplot(
                data=df_clean,
                x=feature_choice,
                hue="target",
                kde=True,
                palette=["#3b82f6", "#ef4444"],
                ax=ax_f1,
            )
            ax_f1.legend(["Heart Disease", "No Disease"])
            ax_f1.set_title(f"Distribution of {feature_choice} by Condition")
            st.pyplot(fig_f1)
            plt.close()

        with col_f2:
            fig_f2, ax_f2 = plt.subplots(figsize=(6, 4))
            sns.boxplot(
                data=df_clean,
                x="target",
                y=feature_choice,
                palette=["#3b82f6", "#ef4444"],
                ax=ax_f2,
            )
            ax_f2.set_xticklabels(["No Disease (0)", "Disease (1)"])
            ax_f2.set_title(f"Boxplot of {feature_choice}")
            st.pyplot(fig_f2)
            plt.close()


# ==============================================================================
# 3. MODEL EVALUATION
# ==============================================================================
elif selected_page == "Model Evaluation":
    st.markdown("## 📈 Rigorous Model Evaluation & Selection Benchmarks")
    st.markdown(
        """
        Following strict machine learning best practices:
        - Candidate models were trained and validated via **5-Fold Stratified Cross-Validation** strictly on the training partition ($N=242$).
        - The final selected pipeline was evaluated **only once** on the held-out test partition ($N=61$).
        """
    )

    selected_model = eval_data["selected_model"]
    st.success(f"🏆 **Selected Model:** **{selected_model}** — {eval_data['selection_rationale']}")

    # CV vs Test Set Comparison Table
    st.markdown("### 📊 Model Performance Comparison Table")

    cv_results = eval_data["cv_results_by_model"]
    test_results = eval_data["test_results_by_model"]

    models_list = list(test_results.keys())
    comparison_records = []
    for m in models_list:
        cv_m = cv_results[m]
        t_m = test_results[m]
        comparison_records.append(
            {
                "Model": m,
                "CV Recall (Train)": f"{cv_m['cv_recall_mean']:.3f} ± {cv_m['cv_recall_std']:.3f}",
                "Test Recall (Sensitivity)": f"{t_m['recall']:.3f}",
                "Test False Negatives (FN)": f"{t_m['false_negatives']} / 28",
                "CV ROC-AUC (Train)": f"{cv_m['cv_roc_auc_mean']:.3f}",
                "Test ROC-AUC": f"{t_m['roc_auc']:.3f}",
                "Test F1-Score": f"{t_m['f1']:.3f}",
                "Test Accuracy": f"{t_m['accuracy']:.3f}",
            }
        )

    comp_df = pd.DataFrame(comparison_records)
    st.dataframe(comp_df, use_container_width=True)

    st.markdown(
        """
        > 💡 **Why Recall and False Negatives Matter:**  
        > In clinical diagnostic screening, a **False Negative** means an individual with coronary disease is erroneously classified as healthy, 
        > potentially foregoing vital diagnostic catheterization or medication. Thus, our selection criterion mandated maximizing validation Recall.
        """
    )

    st.markdown("---")
    st.markdown("### 🎯 Held-out Test Set Visualizations (N=61)")

    col_v1, col_v2 = st.columns(2)

    with col_v1:
        st.markdown(f"#### Confusion Matrix ({selected_model})")
        cm_fig_path = Path("models/figures/confusion_matrix_selected.png")
        if cm_fig_path.exists():
            st.image(str(cm_fig_path), use_container_width=True)
        else:
            st.warning("Confusion matrix plot not found.")

    with col_v2:
        st.markdown("#### Overlaid ROC Curves Comparison")
        roc_fig_path = Path("models/figures/roc_curve_comparison.png")
        if roc_fig_path.exists():
            st.image(str(roc_fig_path), use_container_width=True)
        else:
            st.warning("ROC curves plot not found.")

    st.markdown("---")
    st.markdown("### 🧬 Model Interpretation & Feature Importance")
    st.markdown(
        """
        <div class="disclaimer-banner">
            ⚠️ <strong>Association vs. Causation:</strong> The feature rankings below denote statistical association within the
            historical Cleveland sample. <strong>Statistical importance does not establish biological cause or pathogenesis.</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_i1, col_i2 = st.columns(2)

    with col_i1:
        st.markdown("#### Random Forest Feature Importance (MDI)")
        rf_path = Path("models/figures/rf_feature_importance.png")
        if rf_path.exists():
            st.image(str(rf_path), use_container_width=True)

    with col_i2:
        st.markdown("#### Permutation Importance on Held-out Test Data")
        perm_path = Path("models/figures/permutation_importance.png")
        if perm_path.exists():
            st.image(str(perm_path), use_container_width=True)


# ==============================================================================
# 4. HEALTH RISK PREDICTION
# ==============================================================================
elif selected_page == "Health Risk Prediction":
    st.markdown("## 🩺 Single-Patient Risk Stratification")
    st.markdown(
        """
        Enter standard non-invasive clinical and demographic parameters below to compute an experimental
        model-estimated health risk assessment.
        """
    )

    st.markdown(
        """
        <div class="disclaimer-banner">
            ⚠️ <strong>Clinical Notice:</strong> This prediction is a probabilistic estimate from a machine learning model
            trained on historical research data. It is <strong>NOT</strong> an official medical diagnostic test and should never
            replace consultation with a licensed cardiologist.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Patient Preset Selector
    st.markdown("#### ⚡ Quick-Fill Clinical Demo Presets")
    preset = st.radio(
        "Load sample profile:",
        ["Custom Input", "Typical Low-Risk Profile", "High-Risk Cardiac Profile"],
        horizontal=True,
    )

    # Preset defaults
    if preset == "Typical Low-Risk Profile":
        def_age, def_sex, def_cp = 42, 0, 2
        def_trestbps, def_chol, def_fbs = 118, 195, 0
        def_restecg, def_thalach, def_exang = 0, 168, 0
        def_oldpeak, def_slope, def_ca, def_thal = 0.0, 1, 0, 3
    elif preset == "High-Risk Cardiac Profile":
        def_age, def_sex, def_cp = 64, 1, 4
        def_trestbps, def_chol, def_fbs = 160, 288, 1
        def_restecg, def_thalach, def_exang = 2, 115, 1
        def_oldpeak, def_slope, def_ca, def_thal = 2.8, 2, 2, 7
    else:
        def_age, def_sex, def_cp = 54, 1, 3
        def_trestbps, def_chol, def_fbs = 130, 240, 0
        def_restecg, def_thalach, def_exang = 0, 150, 0
        def_oldpeak, def_slope, def_ca, def_thal = 1.0, 1, 0, 3

    # Input Form
    with st.form("risk_prediction_form"):
        st.markdown("### Patient Clinical Attributes")

        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.number_input("Age (years)", min_value=18, max_value=100, value=def_age, step=1)
            sex = st.selectbox(
                "Biological Sex",
                options=[0, 1],
                index=0 if def_sex == 0 else 1,
                format_func=lambda x: "Female (0)" if x == 0 else "Male (1)",
            )
            cp = st.selectbox(
                "Chest Pain Type (cp)",
                options=[1, 2, 3, 4],
                index=[1, 2, 3, 4].index(def_cp),
                format_func=lambda x: {
                    1: "1: Typical Angina",
                    2: "2: Atypical Angina",
                    3: "3: Non-Anginal Pain",
                    4: "4: Asymptomatic",
                }[x],
            )
            trestbps = st.number_input(
                "Resting Blood Pressure (mm Hg)",
                min_value=80,
                max_value=240,
                value=def_trestbps,
                step=1,
                help="Resting blood pressure on hospital admission.",
            )

        with c2:
            chol = st.number_input(
                "Serum Cholestoral (mg/dl)",
                min_value=100,
                max_value=600,
                value=def_chol,
                step=1,
                help="Serum cholesterol in mg/dl.",
            )
            fbs = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dl (fbs)",
                options=[0, 1],
                index=0 if def_fbs == 0 else 1,
                format_func=lambda x: "False (0: <= 120 mg/dl)" if x == 0 else "True (1: > 120 mg/dl)",
            )
            restecg = st.selectbox(
                "Resting ECG Results (restecg)",
                options=[0, 1, 2],
                index=[0, 1, 2].index(def_restecg),
                format_func=lambda x: {
                    0: "0: Normal",
                    1: "1: ST-T Wave Abnormality",
                    2: "2: Left Ventricular Hypertrophy (Estes' criteria)",
                }[x],
            )
            thalach = st.number_input(
                "Max Heart Rate Achieved (bpm)",
                min_value=60,
                max_value=230,
                value=def_thalach,
                step=1,
                help="Maximum heart rate achieved during exercise stress testing.",
            )

        with c3:
            exang = st.selectbox(
                "Exercise Induced Angina (exang)",
                options=[0, 1],
                index=0 if def_exang == 0 else 1,
                format_func=lambda x: "No (0)" if x == 0 else "Yes (1)",
            )
            oldpeak = st.number_input(
                "ST Depression (oldpeak in mm)",
                min_value=0.0,
                max_value=7.0,
                value=float(def_oldpeak),
                step=0.1,
                help="ST depression induced by exercise relative to rest.",
            )
            slope = st.selectbox(
                "Peak Exercise ST Slope",
                options=[1, 2, 3],
                index=[1, 2, 3].index(def_slope),
                format_func=lambda x: {
                    1: "1: Upsloping",
                    2: "2: Flat",
                    3: "3: Downsloping",
                }[x],
            )
            ca = st.selectbox(
                "Major Vessels by Fluoroscopy (ca)",
                options=[0, 1, 2, 3],
                index=[0, 1, 2, 3].index(def_ca),
                help="Number of major vessels (0-3) colored by fluoroscopy.",
            )
            thal = st.selectbox(
                "Thallium Scintigraphy (thal)",
                options=[3, 6, 7],
                index=[3, 6, 7].index(def_thal),
                format_func=lambda x: {
                    3: "3: Normal",
                    6: "6: Fixed Defect",
                    7: "7: Reversible Defect",
                }[x],
            )

        submitted = st.form_submit_button("⚡ Compute Risk Estimate", use_container_width=True)

    if submitted:
        input_data = pd.DataFrame(
            [
                {
                    "age": float(age),
                    "sex": float(sex),
                    "cp": float(cp),
                    "trestbps": float(trestbps),
                    "chol": float(chol),
                    "fbs": float(fbs),
                    "restecg": float(restecg),
                    "thalach": float(thalach),
                    "exang": float(exang),
                    "oldpeak": float(oldpeak),
                    "slope": float(slope),
                    "ca": float(ca),
                    "thal": float(thal),
                }
            ]
        )

        try:
            prediction = pipeline.predict(input_data)[0]
            probability = (
                pipeline.predict_proba(input_data)[0, 1]
                if hasattr(pipeline, "predict_proba")
                else None
            )

            if prediction == 1:
                st.markdown(
                    f"""
                    <div class="result-box-positive">
                        <h3>⚠️ Elevated Health Risk Indicated</h3>
                        <p style="font-size: 1.1rem; margin-top: 8px;">
                            The trained model identifies patterns consistent with the <strong>presence of heart disease</strong>
                            in historical patient records.
                        </p>
                        <p><strong>Model Output Score:</strong> {probability * 100:.1f}% probabilistic propensity</p>
                        <hr style="border-color: #fca5a5;">
                        <p style="font-size: 0.9rem; margin-bottom: 0;">
                            <strong>Interpretation Note:</strong> This probability is an uncalibrated classification confidence score.
                            It reflects mathematical similarity to positive records in the 1989 Cleveland dataset, <em>not</em> an absolute clinical risk percentage.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-box-negative">
                        <h3>✅ Low Health Risk Indicated</h3>
                        <p style="font-size: 1.1rem; margin-top: 8px;">
                            The trained model associates this clinical profile with <strong>absence of significant heart disease</strong>
                            (&lt;50% diameter stenosis in historical dataset).
                        </p>
                        <p><strong>Model Output Score:</strong> {(1 - probability) * 100:.1f}% negative classification confidence ({probability * 100:.1f}% risk score)</p>
                        <hr style="border-color: #86efac;">
                        <p style="font-size: 0.9rem; margin-bottom: 0;">
                            <strong>Interpretation Note:</strong> A negative classification does not guarantee the absence of coronary conditions.
                            Routine cardiovascular checkups and consultation with medical doctors remain essential.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        except Exception as e:
            st.error(f"Inference error: {e}")


# ==============================================================================
# 5. ABOUT PAGE
# ==============================================================================
elif selected_page == "About":
    st.markdown("## 📖 About the Project")

    st.markdown(
        """
        ### 🎯 Project Objectives & Academic Context
        This application represents a comprehensive, end-to-end Major Project for **B.Tech CSE (Artificial Intelligence & Machine Learning)**.
        The goal is to demonstrate:
        - Rigorous clinical data preprocessing and leakage prevention.
        - Reproducible stratified cross-validation and hyperparameter stability.
        - Objective model selection emphasizing **Recall** and False Negative minimization over naive accuracy.
        - Academic interpretability distinguishing correlation from medical causation.
        - Modern, accessible deployment using Streamlit and Scikit-learn pipelines.

        ---
        ### 🛠️ Technology Stack
        - **Core Language:** Python 3.13
        - **Data Manipulation:** Pandas & NumPy
        - **Machine Learning:** Scikit-learn (Pipelines, ColumnTransformer, Imputers, Classifiers)
        - **Model Persistence:** Joblib
        - **Data Visualizations:** Matplotlib & Seaborn
        - **Web Application:** Streamlit

        ---
        ### ⚠️ Dataset Limitations & Historical Biases
        1. **Sample Size:** The Cleveland database contains only 303 patient records. While widely recognized as a benchmark, small samples introduce statistical variance.
        2. **Demographic Skew:** Collected in 1989 at the Cleveland Clinic Foundation with a predominantly male cohort (67.9% male). Performance may not generalize across diverse global populations or female cardiovascular phenotypes.
        3. **Missing Biomarkers:** Does not incorporate modern high-sensitivity cardiac troponins, coronary calcium scores (CAC), or genetic factors.
        4. **Class Definition:** Heart disease is strictly defined as $\ge 50\%$ diameter stenosis on invasive angiography, which does not encompass microvascular angina or heart failure with preserved ejection fraction.

        ---
        ### ⚖️ Ethical & Responsible AI Considerations
        - **No Replacement for Doctors:** AI classification models are decision-support and academic exploration tools only.
        - **Privacy Preserving:** No personally identifiable information (PII) is stored or transmitted.
        - **Transparent Uncertainty:** Model output scores are uncalibrated algorithmic confidence estimates, not certified diagnostic probabilities.
        """
    )
