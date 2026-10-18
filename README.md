# Health Risk Prediction System

**A Clinical Machine Learning Framework for Preventive Heart Disease Risk Stratification**  
*Major Project — B.Tech Computer Science and Engineering (Artificial Intelligence & Machine Learning)*

---

## 📌 Project Overview
The **Health Risk Prediction System** is an end-to-end educational machine learning solution designed to identify patterns associated with cardiovascular disease risk from patient clinical records. Cardiovascular conditions remain the single largest cause of mortality globally. By exploring non-invasive demographic, electrocardiographic, and exercise-induced biomarkers, this project demonstrates how machine learning models can be trained reproducibly, evaluated with clinical rigor (prioritizing **Recall** to minimize false negatives), and interpreted responsibly without making unsupported claims of clinical diagnosis.

---

## 🎯 Problem Statement & Objectives
- **Problem Statement:** Early identification of cardiovascular risk factors is essential for preventative cardiology. However, naive machine learning implementations frequently suffer from data leakage, rely solely on overall accuracy, and treat classification probabilities as verified diagnostic certainties.
- **Project Objectives:**
  1. Ingest and preprocess the benchmark **UCI Heart Disease (Cleveland)** dataset with strict prevention of data leakage.
  2. Implement an automated Scikit-learn `ColumnTransformer` feature pipeline handling numerical scaling and categorical encoding.
  3. Train and benchmark three classification algorithms: **Logistic Regression**, **Decision Tree Classifier**, and **Random Forest Classifier**.
  4. Perform **5-Fold Stratified Cross-Validation** on training data to select the optimal model using a clinically motivated criterion that minimizes False Negatives (maximizes Recall).
  5. Evaluate the selected model on a held-out test partition ($N=61$) only after selection.
  6. Interpret feature associations using Random Forest MDI importance and permutation importance, explicitly separating correlation from medical causation.
  7. Deploy a modern, interactive **Streamlit web application** providing intuitive risk assessments, patient presets, and transparent disclaimers.

---

## 📦 Minimum Viable Product (MVP) Features
- **Local Data Ingestion:** Automated loading and schema verification of the Cleveland dataset.
- **Exploratory Data Analysis:** Comprehensive analysis of class distributions, missing value patterns, and continuous biomarker trends.
- **Encapsulated Preprocessing:** Median imputation and `StandardScaler` for continuous features; frequent imputation and `OneHotEncoder` for clinical categorical features.
- **Reproducible Partitioning:** Stratified 80/20 train/test split with fixed random seed (`random_state=42`).
- **Clinical Model Selection:** Objective selection prioritizing Cross-Validation Recall to reduce missed disease cases.
- **Joblib Model Persistence:** Complete Scikit-learn pipeline saved to `models/best_model_pipeline.joblib`.
- **Interactive Streamlit UI:** Complete multi-page interface covering Home, Dataset Overview, Model Evaluation, Health Risk Prediction, and About.
- **Clear Ethical Disclaimers:** Explicit notifications that predictions are educational algorithmic estimates, not clinical diagnoses.

---

## 🗃️ Dataset Source & Provenance
- **Dataset:** Heart Disease Dataset (Cleveland Clinic Foundation)
- **Repository:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Direct Source URL:** `https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data`
- **Total Records:** 303 patient instances
- **Total Attributes:** 14 (13 clinical predictors + 1 target variable `num`)

### Academic Citation
> Detrano, R., Janosi, A., Steinbrunn, W., Pfisterer, M., Schmid, J., Sandhu, S., Gopalakrishna, K., Belle, C., & Groves, R. (1989). *International application of a new probability algorithm for the diagnosis of coronary artery disease*. **American Journal of Cardiology**, 64(5), 304–310.

---

## 📋 Input Attributes & Target Definition

| Variable | Description | Type | Clinical Domain Values |
| :--- | :--- | :--- | :--- |
| `age` | Age in years | Numerical | 29 – 77 years |
| `sex` | Biological sex | Binary | 1 = Male; 0 = Female |
| `cp` | Chest pain type | Categorical | 1: Typical Angina, 2: Atypical Angina, 3: Non-Anginal, 4: Asymptomatic |
| `trestbps` | Resting blood pressure | Numerical | 94 – 200 mm Hg (on admission) |
| `chol` | Serum cholestoral | Numerical | 126 – 564 mg/dl |
| `fbs` | Fasting blood sugar > 120 mg/dl | Binary | 1 = True; 0 = False |
| `restecg` | Resting electrocardiogram | Categorical | 0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy |
| `thalach` | Maximum heart rate achieved | Numerical | 71 – 202 bpm (during exercise test) |
| `exang` | Exercise-induced angina | Binary | 1 = Yes; 0 = No |
| `oldpeak` | ST depression induced by exercise | Numerical | 0.0 – 6.2 mm relative to rest |
| `slope` | Slope of peak exercise ST segment | Categorical | 1: Upsloping, 2: Flat, 3: Downsloping |
| `ca` | Major vessels colored by fluoroscopy | Discrete | 0, 1, 2, 3 vessels (4 missing values coded `?`) |
| `thal` | Thallium scintigraphy defect | Categorical | 3.0: Normal, 6.0: Fixed defect, 7.0: Reversible defect (2 missing values coded `?`) |
| `num` | Angiographic disease severity | Target | 0: <50% stenosis; 1–4: >50% stenosis |

### Target Binarization
The raw target `num` was transformed into a binary outcome:
- **`target = 0` (No Heart Disease):** 164 instances (54.1%) — $<50\%$ diameter narrowing.
- **`target = 1` (Heart Disease Detected):** 139 instances (45.9%) — $\ge 50\%$ diameter narrowing (original values 1–4).

The target variable is excluded from feature matrix $X$ before preprocessing.

---

## 🛠️ Technology Stack
- **Programming Language:** Python 3.13
- **Data Engineering:** Pandas 2.x, NumPy 2.x
- **Machine Learning:** Scikit-learn 1.9+ (Pipelines, ColumnTransformer, Metrics, Imputers)
- **Model Serialization:** Joblib 1.6+
- **Data Visualization:** Matplotlib 3.11+, Seaborn 0.13+
- **Web Interface:** Streamlit 1.66+
- **Version Control:** Git & GitHub

---

## 📂 Project Structure
```text
Healt-Risk-Prediction-System/
├── data/
│   ├── README.md                      # Dataset documentation and provenance
│   ├── raw/
│   │   └── heart_disease_raw.csv      # Raw downloaded Cleveland CSV (303 rows)
│   └── processed/
│       └── heart_disease_clean.csv    # Cleaned dataset with binarized target
├── models/
│   ├── .gitkeep                       # Directory preservation
│   ├── best_model_pipeline.joblib     # Saved fitted Scikit-learn pipeline
│   ├── model_metadata.json            # Training configurations and CV metrics
│   ├── test_evaluation_results.json   # Held-out test evaluation benchmarks
│   └── figures/
│       ├── confusion_matrix_selected.png
│       ├── roc_curve_comparison.png
│       ├── rf_feature_importance.png
│       └── permutation_importance.png
├── notebooks/
│   └── exploratory_analysis.ipynb     # Jupyter notebook for exploratory data analysis
├── src/
│   ├── __init__.py                    # Source package indicator
│   ├── data_preprocessing.py          # Schema validation, ingestion, and ColumnTransformer
│   ├── train_model.py                 # Cross-validation, model selection, and training
│   └── evaluate_model.py              # Test set evaluation, metric generation, and plotting
├── app.py                             # Interactive Streamlit multi-page application
├── requirements.txt                   # Pinned project dependencies
├── .gitignore                         # Standard git exclusion rules
└── README.md                          # Comprehensive project documentation
```

---

## ⚙️ Data Preprocessing & Leakage-Free Workflow
To guarantee zero data leakage between training and testing splits:
1. **Stratified Split First:** The dataset is split into training ($80\%$, $N=242$) and testing ($20\%$, $N=61$) partitions before fitting any statistical transformer.
2. **ColumnTransformer Construction:**
   - **Numerical Pipeline:** `SimpleImputer(strategy='median')` followed by `StandardScaler()`.
   - **Categorical Pipeline:** `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(categories=known_categories, drop='first', sparse_output=False, handle_unknown='ignore')`.
   - **Binary/Discrete Pipeline:** `SimpleImputer(strategy='most_frequent')` to resolve the 4 missing entries in `ca`.
3. **Pipeline Encapsulation:** Every candidate model is coupled directly to the preprocessor inside a Scikit-learn `Pipeline`, guaranteeing that fold validation and test sets are transformed strictly using training statistics.

---

## 🔬 Model Training & Clinical Selection Methodology

### Documented Selection Criterion
In clinical risk prediction, a **False Negative** (classifying a cardiac patient as healthy) can lead to omitted treatment and severe health events. Conversely, a **False Positive** merely leads to non-invasive confirmatory testing. Consequently:
- **Primary Metric:** Highest Mean Cross-Validation **Recall** on the training partition ($N=242$).
- **Tie-Breaker:** Highest Mean Cross-Validation **ROC-AUC** followed by **F1-Score**.
- **Accuracy Safeguard:** Accuracy was monitored to avoid trivial majority-class classifiers, but was never used as the sole deciding factor.

### Actual Model Results

#### 1. 5-Fold Stratified Cross-Validation on Training Set ($N=242$)
| Candidate Model | Mean CV Recall | Mean CV ROC-AUC | Mean CV F1-Score | Mean CV Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (Selected)** | **0.7834 ± 0.036** | **0.9089 ± 0.043** | **0.8214 ± 0.024** | **0.8429 ± 0.022** |
| **Random Forest** | 0.7652 ± 0.047 | 0.8958 ± 0.035 | 0.7874 ± 0.036 | 0.8100 ± 0.032 |
| **Decision Tree** | 0.7202 ± 0.090 | 0.8239 ± 0.057 | 0.7323 ± 0.054 | 0.7606 ± 0.043 |

*Logistic Regression achieved the highest mean cross-validation recall (78.34%) and highest ROC-AUC (0.9089), making it the optimal model under our clinical validation protocol.*

#### 2. Final Evaluation on Held-Out Test Set ($N=61$, 33 Negative, 28 Positive)
| Model | Test Accuracy | Test Recall (Sensitivity) | Test Precision | Test F1-Score | Test ROC-AUC | False Negatives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Selected)** | **85.25%** | **89.29%** | **80.65%** | **84.75%** | **0.9567** | **3 / 28** |
| **Random Forest** | 90.16% | 92.86% | 86.67% | 89.66% | 0.9665 | 2 / 28 |
| **Decision Tree** | 78.69% | 67.86% | 82.61% | 74.51% | 0.8496 | 9 / 28 |

#### Confusion Matrix for Selected Model (Logistic Regression)
- **True Negatives (TN):** 27
- **False Positives (FP):** 6
- **False Negatives (FN):** 3
- **True Positives (TP):** 25
- **Sensitivity / Recall:** $25 / (25 + 3) = 89.29\%$
- **Specificity:** $27 / (27 + 6) = 81.82\%$

---

## 🧬 Model Interpretation & Explainability
- **Random Forest MDI Importance:** Evaluates Gini impurity reduction. Identified `ca` (fluoroscopy vessels), `thal` (thallium defect), `oldpeak` (ST depression), and `thalach` (max heart rate) as top predictive attributes.
- **Permutation Importance on Test Set:** Evaluates performance degradation when values of an attribute are randomly shuffled. Re-shuffling `ca` and `thal` caused the steepest drop in test ROC-AUC.
- **Critical Distinction:** **Association is NOT causation.** A high feature importance value indicates strong mathematical predictive utility within this 1989 cohort, not biological etiology or cardiac pathology.

---

## 🚀 Setup & Execution Guide

### 1. Prerequisites
Ensure you have **Python 3.10+** (tested on Python 3.13) and `git` installed.

### 2. Environment Setup
Clone the repository and initialize a virtual environment:
```bash
# Clone the repository
git clone https://github.com/your-username/Healt-Risk-Prediction-System.git
cd Healt-Risk-Prediction-System

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Step-by-Step Pipeline Execution
Execute the pipeline stages sequentially:

```bash
# Step 1: Download raw data, validate schema, and produce clean CSV
python src/data_preprocessing.py

# Step 2: Run 5-Fold Stratified Cross-Validation, select model, fit on train set, and save
python src/train_model.py

# Step 3: Evaluate on held-out test set, compute metrics, and generate visual figures
python src/evaluate_model.py
```

### 4. Launch the Streamlit Web Application
Run the interactive interface locally:
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 🖼️ Application Interface & Screenshots

| Section | Description | Status |
| :--- | :--- | :---: |
| **Home Page** | Project scope, problem statement, workflow overview, and clinical disclaimer. | *Active* |
| **Dataset Overview** | Data table, missing-value audit, target proportions, and biomarker histograms. | *Active* |
| **Model Evaluation** | CV benchmark table, test metrics, confusion matrix, and ROC curves. | *Active* |
| **Health Risk Prediction** | Interactive clinical form with quick-fill presets and probability estimates. | *Active* |
| **About Page** | Educational objectives, technology stack, ethical AI, and dataset limitations. | *Active* |

*(Screenshots can be captured directly from your local browser session at `http://localhost:8501` and placed in `models/figures/`)*

---

## ⚖️ Responsible AI, Ethical Considerations & Limitations
1. **Historical Cohort Bias:** The Cleveland dataset was gathered in 1989 across 303 patients, primarily male ($67.9\%$). Patterns may not represent female cardiac symptom manifestations or modern diverse populations.
2. **Small Sample Uncertainty:** With $N=61$ in the test set, each individual sample accounts for $\approx 1.6\%$ to $3.5\%$ shifts in metrics.
3. **Absence of Modern Biomarkers:** Key clinical standards such as high-sensitivity troponin assays, coronary artery calcium (CAC) scoring, and detailed lipid fractions (ApoB, Lp(a)) are absent.
4. **Algorithmic Risk Scores:** Output probabilities are mathematical model confidence scores, not calibrated clinical risk percentages.
5. **No Medical Advice:** The system is an educational simulation. It does not possess medical certification and must never substitute for a licensed physician.

---

## 🔧 Troubleshooting
- **`ModuleNotFoundError: No module named 'src'`:**  
  Run scripts with the virtual environment active from the project root directory, e.g.:  
  `python src/train_model.py` or `python -m src.train_model`.
- **Streamlit Port Already in Use:**  
  If port 8501 is occupied, specify an alternative port:  
  `streamlit run app.py --server.port 8502`
- **Missing Dataset:**  
  If offline, ensure `data/raw/heart_disease_raw.csv` is present. `data_preprocessing.py` will automatically attempt to fetch it if internet access is available.

---

## 🔮 Future Improvements
- Implement calibration curves and Platt scaling / isotonic regression for true probabilistic calibration.
- Integrate SHAP (SHapley Additive exPlanations) for local patient-level explanation waterfall plots.
- Expand to multi-center datasets (combining Hungarian, Swiss, and Long Beach cohorts after schema harmonization).
- Package application into a Docker container for cross-platform portability.

---

## 📤 GitHub Upload Guidance
To publish this project to GitHub:
```bash
git init
git add .
git commit -m "feat: complete Health Risk Prediction System MVP with Streamlit app and evaluation"
git branch -M main
git remote add origin https://github.com/<your-username>/Healt-Risk-Prediction-System.git
git push -u origin main
```

---

## 📜 Academic Project Disclaimer
This project is submitted in partial fulfillment of the requirements for the **B.Tech Degree in Computer Science & Engineering (Artificial Intelligence and Machine Learning)**. All statistical metrics and pipeline evaluations reflect genuine experimental runs executed on the Cleveland dataset.
