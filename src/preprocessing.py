"""Data loading and leakage-safe preprocessing for the IBM Telco dataset."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "customer_churn.csv"
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the public CSV and apply transparent, deterministic cleaning."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].str.strip()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.drop_duplicates().reset_index(drop=True)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0}).astype("int64")

    # The blank TotalCharges cells occur for brand-new customers. The model
    # imputes the training median without looking at the held-out test set.
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return model features and a binary churn target."""
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    return df[feature_columns].copy(), df[TARGET_COLUMN].copy()


def build_preprocessor() -> ColumnTransformer:
    """Build the preprocessing used inside every model pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def model_input_from_form(values: dict) -> pd.DataFrame:
    """Convert prediction form values into the model's expected one-row frame."""
    return pd.DataFrame(
        [
            {
                "SeniorCitizen": int(values["SeniorCitizen"]),
                "tenure": float(values["tenure"]),
                "MonthlyCharges": float(values["MonthlyCharges"]),
                "TotalCharges": float(values["TotalCharges"]),
                **{column: values[column] for column in CATEGORICAL_FEATURES},
            }
        ]
    )[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
