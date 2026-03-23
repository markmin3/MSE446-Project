#!/usr/bin/env python3
"""
Baseline models for the Spotify project.

Implements:
1) Logistic Regression for genre classification
2) Linear Regression for popularity prediction

Dataset expected: spotify_balanced.csv
"""

from __future__ import annotations

import argparse

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_DATASET = "spotify_balanced.csv"

# Audio + metadata columns available in the remapped balanced dataset
FEATURE_COLUMNS = [
    "duration_ms",
    "explicit",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
]

CLASS_TARGET = "track_genre"
REGRESSION_TARGET = "popularity"


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features and keep only required columns."""
    required = FEATURE_COLUMNS + [CLASS_TARGET, REGRESSION_TARGET]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    data = df[required].copy()

    # Ensure explicit is numeric (0/1)
    if data["explicit"].dtype == bool:
        data["explicit"] = data["explicit"].astype(int)
    else:
        data["explicit"] = (
            data["explicit"].astype(str).str.lower().map({"true": 1, "false": 0})
        )
        data["explicit"] = data["explicit"].fillna(0).astype(int)

    return data.dropna()


def run_logistic_regression(data: pd.DataFrame, test_size: float, random_state: int) -> None:
    """Train and evaluate baseline multiclass logistic regression."""
    x = data[FEATURE_COLUMNS]
    y = data[CLASS_TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), FEATURE_COLUMNS),
        ]
    )

    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ]
    )

    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)

    print("\n=== Logistic Regression (Genre Classification) ===")
    print(f"Train samples: {len(x_train)} | Test samples: {len(x_test)}")
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, preds, digits=4))


def run_linear_regression(data: pd.DataFrame, test_size: float, random_state: int) -> None:
    """Train and evaluate baseline linear regression for popularity."""
    x = data[FEATURE_COLUMNS]
    y = data[REGRESSION_TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), FEATURE_COLUMNS),
        ]
    )

    reg = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LinearRegression()),
        ]
    )

    reg.fit(x_train, y_train)
    preds = reg.predict(x_test)

    rmse = mean_squared_error(y_test, preds) ** 0.5
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("\n=== Linear Regression (Popularity Prediction) ===")
    print(f"Train samples: {len(x_train)} | Test samples: {len(x_test)}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R^2:  {r2:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline ML models.")
    parser.add_argument(
        "--data",
        type=str,
        default=DEFAULT_DATASET,
        help=f"Path to dataset CSV (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio (default: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    data = build_feature_table(df)

    print(f"Loaded dataset: {args.data}")
    print(f"Usable rows after cleaning: {len(data)}")
    print(f"Genres: {data[CLASS_TARGET].nunique()}")

    run_logistic_regression(data, test_size=args.test_size, random_state=args.random_state)
    run_linear_regression(data, test_size=args.test_size, random_state=args.random_state)


if __name__ == "__main__":
    main()
