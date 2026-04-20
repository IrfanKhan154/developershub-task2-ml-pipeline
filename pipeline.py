"""End-to-end ML pipeline for Telco churn prediction."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.error import HTTPError, URLError

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_telco_churn_dataset(data_path: str | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Load the Telco churn dataset and return features and target."""
    # Prefer a local CSV when available to avoid network dependency.
    candidate_paths = []
    if data_path:
        candidate_paths.append(Path(data_path))
    candidate_paths.append(Path("WA_Fn-UseC_-Telco-Customer-Churn.csv"))

    df = None
    for path in candidate_paths:
        if path.exists():
            df = pd.read_csv(path)
            break

    # Fall back to OpenML if local dataset is not found.
    if df is None:
        try:
            data = fetch_openml(name="Telco-Customer-Churn", version=1, as_frame=True)
            df = data.frame.copy()
        except (URLError, HTTPError) as exc:
            raise RuntimeError(
                "Could not load Telco churn dataset. Provide a local CSV file "
                "(e.g., WA_Fn-UseC_-Telco-Customer-Churn.csv) or run with network access."
            ) from exc

    # Remove ID-like column if present because it has no predictive value.
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Separate target from features and normalize target labels to 0/1.
    y = df["Churn"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    if y.isna().any():
        raise ValueError(
            "Target column 'Churn' contains unexpected values. "
            "Expected only 'Yes'/'No' entries."
        )
    x = df.drop(columns=["Churn"])

    # Convert known numeric-like columns that can be stored as strings.
    if "TotalCharges" in x.columns:
        x["TotalCharges"] = pd.to_numeric(x["TotalCharges"], errors="coerce")

    return x, y


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    """Build a preprocessor that scales numeric columns and one-hot encodes categorical columns."""
    numeric_features = x.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = x.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def train_and_evaluate(data_path: str | None = None) -> None:
    """Train Logistic Regression and Random Forest with GridSearchCV, evaluate, and save best model."""
    x, y = load_telco_churn_dataset(data_path=data_path)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(x_train)

    # Define model pipelines and their hyperparameter grids.
    models = {
        "logistic_regression": (
            Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
                ]
            ),
            {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__solver": ["lbfgs"],
            },
        ),
        "random_forest": (
            Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", RandomForestClassifier(random_state=42)),
                ]
            ),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 10, 20],
                "classifier__min_samples_split": [2, 5],
            },
        ),
    }

    best_name = None
    best_estimator = None
    best_accuracy = -1.0

    for model_name, (pipeline, param_grid) in models.items():
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="accuracy",
            cv=5,
            n_jobs=-1,
        )
        grid_search.fit(x_train, y_train)

        y_pred = grid_search.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"{model_name} accuracy: {accuracy:.4f}")
        print(f"{model_name} best params: {grid_search.best_params_}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_name = model_name
            best_estimator = grid_search.best_estimator_

    # Persist the best-performing trained pipeline to disk.
    joblib.dump(best_estimator, "best_model.joblib")
    print(f"Best model: {best_name} (accuracy={best_accuracy:.4f})")
    print("Saved model to best_model.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate Telco churn ML pipeline.")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Optional local path to Telco churn CSV file.",
    )
    args = parser.parse_args()
    train_and_evaluate(data_path=args.data_path)
