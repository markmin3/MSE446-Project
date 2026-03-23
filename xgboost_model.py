#!/usr/bin/env python3
from __future__ import annotations

import argparse

from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from ml_common import DEFAULT_DATASET, load_clean_data, split_classification


def main() -> None:
    parser = argparse.ArgumentParser(description="XGBoost classifier baseline.")
    parser.add_argument("--data", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    data = load_clean_data(args.data)
    x_train, x_test, y_train_raw, y_test_raw = split_classification(
        data, test_size=args.test_size, random_state=args.random_state
    )

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_raw)
    y_test = encoder.transform(y_test_raw)

    model = XGBClassifier(
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
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    model.fit(x_train, y_train, sample_weight=sample_weight)

    preds = model.predict(x_test)
    acc = accuracy_score(y_test, preds)
    target_names = list(encoder.classes_)

    print("=== XGBoost (Genre Classification) ===")
    print(f"Train: {len(x_train)} | Test: {len(x_test)}")
    print(f"Accuracy: {acc:.4f}")
    print(
        classification_report(
            y_test,
            preds,
            target_names=target_names,
            digits=4,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
