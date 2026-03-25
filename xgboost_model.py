#!/usr/bin/env python3
from __future__ import annotations

import argparse

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

from ml_common import (
    DEFAULT_DATASET,
    load_clean_data,
    split_classification,
    split_regression,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="XGBoost baselines (Classification & Regression).")
    parser.add_argument("--data", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    data = load_clean_data(args.data)

    #Genre Classification
    x_train_c, x_test_c, y_train_raw, y_test_raw = split_classification(
        data, test_size=args.test_size, random_state=args.random_state
    )

    encoder = LabelEncoder()
    y_train_c = encoder.fit_transform(y_train_raw)
    y_test_c = encoder.transform(y_test_raw)

    clf = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_estimators=700,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=args.random_state,
        n_jobs=-1,
    )
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train_c)
    clf.fit(x_train_c, y_train_c, sample_weight=sample_weight)

    preds_c = clf.predict(x_test_c)
    acc = accuracy_score(y_test_c, preds_c)
    target_names = list(encoder.classes_)

    print("=== XGBoost (Genre Classification) ===")
    print(f"Train: {len(x_train_c)} | Test: {len(x_test_c)}")
    print(f"Accuracy: {acc:.4f}")
    print(
        classification_report(
            y_test_c,
            preds_c,
            target_names=target_names,
            digits=4,
            zero_division=0,
        )
    )
    print("\n")

    #Popularity Regression
    x_train_r, x_test_r, y_train_r, y_test_r = split_regression(
        data, test_size=args.test_size, random_state=args.random_state
    )

    reg = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=700,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=args.random_state,
        n_jobs=-1,
    )
    reg.fit(x_train_r, y_train_r)
    preds_r = reg.predict(x_test_r)

    rmse = mean_squared_error(y_test_r, preds_r) ** 0.5
    mae = mean_absolute_error(y_test_r, preds_r)
    r2 = r2_score(y_test_r, preds_r)

    print("=== XGBoost (Popularity Regression) ===")
    print(f"Train: {len(x_train_r)} | Test: {len(x_test_r)}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R^2:  {r2:.4f}")


if __name__ == "__main__":
    main()