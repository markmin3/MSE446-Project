#!/usr/bin/env python3
from __future__ import annotations

import argparse

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from ml_common import DEFAULT_DATASET, load_clean_data, split_classification, split_regression


def main() -> None:
    parser = argparse.ArgumentParser(description="Random Forest baselines.")
    parser.add_argument("--data", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    data = load_clean_data(args.data)

    x_train_c, x_test_c, y_train_c, y_test_c = split_classification(
        data, test_size=args.test_size, random_state=args.random_state
    )
    clf = RandomForestClassifier(
        n_estimators=300, random_state=args.random_state, n_jobs=-1
    )
    clf.fit(x_train_c, y_train_c)
    preds_c = clf.predict(x_test_c)

    print("=== Random Forest (Genre Classification) ===")
    print(f"Train: {len(x_train_c)} | Test: {len(x_test_c)}")
    print(f"Accuracy: {accuracy_score(y_test_c, preds_c):.4f}")
    print(classification_report(y_test_c, preds_c, digits=4))

    x_train_r, x_test_r, y_train_r, y_test_r = split_regression(
        data, test_size=args.test_size, random_state=args.random_state
    )
    reg = RandomForestRegressor(
        n_estimators=300, random_state=args.random_state, n_jobs=-1
    )
    reg.fit(x_train_r, y_train_r)
    preds_r = reg.predict(x_test_r)

    rmse = mean_squared_error(y_test_r, preds_r) ** 0.5
    mae = mean_absolute_error(y_test_r, preds_r)
    r2 = r2_score(y_test_r, preds_r)
    print("=== Random Forest (Popularity Regression) ===")
    print(f"Train: {len(x_train_r)} | Test: {len(x_test_r)}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R^2:  {r2:.4f}")


if __name__ == "__main__":
    main()
