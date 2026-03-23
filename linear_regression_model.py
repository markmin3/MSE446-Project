#!/usr/bin/env python3
from __future__ import annotations

import argparse

from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_common import DEFAULT_DATASET, load_clean_data, split_classification


def main() -> None:
    parser = argparse.ArgumentParser(description="Linear classifier baseline.")
    parser.add_argument("--data", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    data = load_clean_data(args.data)
    x_train, x_test, y_train, y_test = split_classification(
        data, test_size=args.test_size, random_state=args.random_state
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", RidgeClassifier()),
        ]
    )

    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    acc = accuracy_score(y_test, preds)

    print("=== Linear Model (Genre Classification) ===")
    print(f"Train: {len(x_train)} | Test: {len(x_test)}")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, digits=4, zero_division=0))


if __name__ == "__main__":
    main()
