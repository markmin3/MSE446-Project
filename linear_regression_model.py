#!/usr/bin/env python3
from __future__ import annotations

import argparse

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_common import DEFAULT_DATASET, load_clean_data, split_regression


def main() -> None:
    parser = argparse.ArgumentParser(description="Linear regression baseline.")
    parser.add_argument("--data", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    data = load_clean_data(args.data)
    x_train, x_test, y_train, y_test = split_regression(
        data, test_size=args.test_size, random_state=args.random_state
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("reg", LinearRegression()),
        ]
    )

    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("=== Linear Regression (Popularity) ===")
    print(f"Train: {len(x_train)} | Test: {len(x_test)}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R^2:  {r2:.4f}")


if __name__ == "__main__":
    main()
