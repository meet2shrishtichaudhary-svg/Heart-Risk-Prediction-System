# UCI Heart Disease Dataset (Cleveland Subset)

## Dataset Overview
- **Name:** Heart Disease Dataset (Cleveland Database)
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Official URL:** `https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data`
- **Total Records:** 303 patient records
- **Total Attributes:** 14 (13 clinical/demographic predictive features + 1 target variable `num`)

## Donors & Principal Investigators
1. Hungarian Institute of Cardiology, Budapest: Andras Janosi, M.D.
2. University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
3. University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.

### Academic Citation
> Detrano, R., Janosi, A., Steinbrunn, W., Pfisterer, M., Schmid, J., Sandhu, S., Gopalakrishna, K., Belle, C., & Groves, R. (1989). *International application of a new probability algorithm for the diagnosis of coronary artery disease*. American Journal of Cardiology, 64(5), 304–310.

---

## Attribute Information

| Index | Attribute | Description | Data Type | Permissible / Observed Range |
|-------|-----------|-------------|-----------|------------------------------|
| 1 | `age` | Age in completed years | Continuous | 29 – 77 years |
| 2 | `sex` | Biological sex | Binary | 1 = Male; 0 = Female |
| 3 | `cp` | Chest pain type | Categorical | 1 = Typical angina, 2 = Atypical angina, 3 = Non-anginal pain, 4 = Asymptomatic |
| 4 | `trestbps` | Resting blood pressure (on hospital admission) | Continuous | 94 – 200 mm Hg |
| 5 | `chol` | Serum cholestoral | Continuous | 126 – 564 mg/dl |
| 6 | `fbs` | Fasting blood sugar > 120 mg/dl | Binary | 1 = True; 0 = False |
| 7 | `restecg` | Resting electrocardiographic results | Categorical | 0 = Normal, 1 = ST-T wave abnormality, 2 = Probable/definite left ventricular hypertrophy |
| 8 | `thalach` | Maximum heart rate achieved during exercise | Continuous | 71 – 202 bpm |
| 9 | `exang` | Exercise induced angina | Binary | 1 = Yes; 0 = No |
| 10 | `oldpeak` | ST depression induced by exercise relative to rest | Continuous | 0.0 – 6.2 mm |
| 11 | `slope` | Slope of the peak exercise ST segment | Categorical | 1 = Upsloping, 2 = Flat, 3 = Downsloping |
| 12 | `ca` | Number of major vessels (0–3) colored by fluoroscopy | Discrete/Ordinal | 0.0, 1.0, 2.0, 3.0 (4 missing values marked `?`) |
| 13 | `thal` | Thallium stress scintigraphy | Categorical | 3.0 = Normal, 6.0 = Fixed defect, 7.0 = Reversible defect (2 missing values marked `?`) |
| 14 | `num` | Diagnosis of heart disease (angiographic disease status) | Target (Ordinal) | 0: <50% diameter narrowing (absence); 1–4: >50% narrowing (presence) |

---

## Target Transformation (Binarization)
The raw dataset contains `num` values ranging from 0 to 4:
- `num == 0`: 164 instances (54.1%) — Absence of significant heart disease
- `num >= 1`: 139 instances (45.9%) — Presence of heart disease (Values: 1: 55, 2: 36, 3: 35, 4: 13)

For the purpose of classification, the target is binarized:
$$ \text{target} = \begin{cases} 0 & \text{if } num = 0 \text{ (No Heart Disease)} \\ 1 & \text{if } num > 0 \text{ (Heart Disease Detected)} \end{cases} $$

The target is never included in the input feature set $X$.

---

## Missing Value Treatment
- `ca` contains 4 missing records (`?`).
- `thal` contains 2 missing records (`?`).
- Total affected rows: 6 out of 303 (1.98%).
- **Treatment Strategy:** To avoid discarding potentially valuable clinical records and prevent data leakage, missing values are imputed strictly within the Scikit-learn Pipeline using the most frequent strategy fitted only on the training split.

## Directory Layout
- `data/raw/heart_disease_raw.csv`: Raw Cleveland dataset with header column names.
- `data/processed/heart_disease_clean.csv`: Cleaned dataset with missing values identified and binary target column added.
