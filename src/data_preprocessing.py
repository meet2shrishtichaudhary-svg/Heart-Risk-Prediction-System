"""Data Preprocessing and Feature Pipeline for Health Risk Prediction System.

This module provides data loading, schema validation, target binarization,
reproducible stratified data splitting, and Scikit-learn ColumnTransformer
pipeline construction.
"""

import os
import urllib.request
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Official UCI Cleveland Database URL
UCI_CLEVELAND_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
)

# Standard 14 column names
COLUMN_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

# Clinical feature groupings
NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]
BINARY_DISCRETE_FEATURES = ["sex", "fbs", "exang", "ca"]

FEATURE_NAMES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + BINARY_DISCRETE_FEATURES


def download_raw_dataset(destination_path: str = "data/raw/heart_disease_raw.csv") -> str:
    """Download official UCI Cleveland dataset if not already present locally."""
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    if os.path.exists(destination_path):
        return destination_path

    print(f"Downloading dataset from UCI repository to {destination_path}...")
    req = urllib.request.Request(
        UCI_CLEVELAND_URL,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )
    with urllib.request.urlopen(req, timeout=15) as response:
        raw_text = response.read().decode("utf-8")

    # Save with header
    lines = [line.strip() for line in raw_text.strip().split("\n") if line.strip()]
    csv_content = ",".join(COLUMN_NAMES) + "\n" + "\n".join(lines) + "\n"

    with open(destination_path, "w", encoding="utf-8") as f:
        f.write(csv_content)

    print(f"Dataset successfully saved with {len(lines)} records.")
    return destination_path


def load_dataset(filepath: str = "data/raw/heart_disease_raw.csv") -> pd.DataFrame:
    """Load the dataset from CSV file and handle missing value representations."""
    if not os.path.exists(filepath):
        # Auto-download if missing
        download_raw_dataset(filepath)

    df = pd.read_csv(filepath, na_values=["?", "NA", "null", ""])

    # If file was loaded without headers, assign them
    if list(df.columns) != COLUMN_NAMES:
        if len(df.columns) == len(COLUMN_NAMES):
            # Check if first row is data
            try:
                float(df.iloc[0, 0])
                df = pd.read_csv(filepath, header=None, names=COLUMN_NAMES, na_values=["?", "NA", "null", ""])
            except ValueError:
                pass

    validate_schema(df)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Validate schema, expected columns, and data integrity."""
    missing_cols = [col for col in COLUMN_NAMES if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset schema mismatch! Missing required columns: {missing_cols}. "
            f"Expected schema: {COLUMN_NAMES}"
        )

    if len(df) == 0:
        raise ValueError("Dataset is empty. At least one record is required.")

    # Validate that target column exists
    if "num" not in df.columns and "target" not in df.columns:
        raise ValueError("Target column ('num' or 'target') not found in dataset.")


def prepare_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract features (X) and binarize target (y).

    Target binarization:
      0 = No heart disease (<50% diameter narrowing)
      1 = Heart disease detected (>=50% diameter narrowing, original num 1-4)
    """
    df_clean = df.copy()

    # Create target column if num is present
    if "num" in df_clean.columns:
        y = (df_clean["num"] > 0).astype(int)
        df_clean["target"] = y
        X = df_clean.drop(columns=["num", "target"], errors="ignore")
    elif "target" in df_clean.columns:
        y = df_clean["target"].astype(int)
        X = df_clean.drop(columns=["target"], errors="ignore")
    else:
        raise ValueError("Target column 'num' not found.")

    # Reorder X columns to canonical order
    X = X[FEATURE_NAMES]

    # Convert numeric types
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform reproducible stratified train-test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def get_preprocessor() -> ColumnTransformer:
    """Construct Scikit-learn ColumnTransformer for preprocessing.

    - Numerical features: Median Imputation -> StandardScaler
    - Categorical features: Most Frequent Imputation -> OneHotEncoder(drop='first', handle_unknown='ignore')
    - Binary/Discrete features: Most Frequent Imputation -> Passthrough
    """
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    known_categories = [
        [1.0, 2.0, 3.0, 4.0],  # cp (1: typical, 2: atypical, 3: non-anginal, 4: asymptomatic)
        [0.0, 1.0, 2.0],        # restecg (0: normal, 1: ST-T wave, 2: hypertrophy)
        [1.0, 2.0, 3.0],        # slope (1: upsloping, 2: flat, 3: downsloping)
        [3.0, 6.0, 7.0],        # thal (3: normal, 6: fixed defect, 7: reversible defect)
    ]

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    categories=known_categories,
                    drop="first",
                    sparse_output=False,
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    bin_discrete_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
            ("bin_disc", bin_discrete_pipeline, BINARY_DISCRETE_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def save_processed_data(
    df: pd.DataFrame, output_path: str = "data/processed/heart_disease_clean.csv"
) -> str:
    """Save cleaned dataset with binarized target for analysis and documentation."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean = df.copy()
    if "num" in df_clean.columns and "target" not in df_clean.columns:
        df_clean["target"] = (df_clean["num"] > 0).astype(int)
    df_clean.to_csv(output_path, index=False)
    print(f"Saved processed dataset to {output_path}")
    return output_path


if __name__ == "__main__":
    raw_path = download_raw_dataset()
    raw_df = load_dataset(raw_path)
    print(f"Loaded raw dataset with shape: {raw_df.shape}")
    save_processed_data(raw_df)
    X, y = prepare_features_and_target(raw_df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Train features shape: {X_train.shape}, Test features shape: {X_test.shape}")
    print(f"Train class distribution:\n{y_train.value_counts(normalize=True)}")
    print(f"Test class distribution:\n{y_test.value_counts(normalize=True)}")
