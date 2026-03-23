#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
)
from sklearn.model_selection import ParameterSampler, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from ml_common import CLASS_TARGET, DEFAULT_DATASET, FEATURE_COLUMNS, load_clean_data


def build_model(random_state: int) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=7,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=0.2,
        reg_alpha=0.05,
        reg_lambda=1.2,
        max_delta_step=1,
        random_state=random_state,
        n_jobs=-1,
    )


def top3_scorer(
    estimator: XGBClassifier,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Scorer for RandomizedSearchCV that optimizes Top-3 accuracy."""
    probs = estimator.predict_proba(x_val)
    labels = list(range(probs.shape[1]))
    return top_k_accuracy_score(
        y_val, probs, k=3, labels=labels, sample_weight=sample_weight
    )


def tune_model(
    base_model: XGBClassifier,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    random_state: int,
    n_iter_stage1: int,
    n_iter_stage2: int,
) -> XGBClassifier:
    # Broad stage: explore a wide space
    stage1_space: dict[str, list[Any]] = {
        "n_estimators": [700, 1000, 1300, 1600, 2000],
        "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.07],
        "max_depth": [4, 5, 6, 7, 8, 9, 10],
        "min_child_weight": [1, 2, 4, 6, 8, 10],
        "subsample": [0.65, 0.75, 0.85, 0.95, 1.0],
        "colsample_bytree": [0.65, 0.75, 0.85, 0.95, 1.0],
        "gamma": [0.0, 0.1, 0.2, 0.5, 1.0, 2.0],
        "reg_alpha": [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
        "reg_lambda": [0.5, 0.8, 1.0, 1.3, 1.8, 2.5, 4.0],
        "max_delta_step": [0, 1, 2, 3, 5],
    }
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    stage1 = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=stage1_space,
        n_iter=n_iter_stage1,
        scoring=top3_scorer,
        n_jobs=1,
        cv=3,
        random_state=random_state,
        verbose=1,
    )
    stage1.fit(x_train, y_train, sample_weight=sample_weight)
    print(f"Best CV Top-3 (stage 1): {stage1.best_score_:.4f}")
    print(f"Best params (stage 1): {stage1.best_params_}")

    # Narrow stage: sample near the stage1 best params
    p = stage1.best_params_
    stage2_space: dict[str, list[Any]] = {
        "n_estimators": sorted(set([max(400, p["n_estimators"] - 300), p["n_estimators"], p["n_estimators"] + 300])),
        "learning_rate": sorted(set([max(0.005, p["learning_rate"] * 0.7), p["learning_rate"], min(0.2, p["learning_rate"] * 1.3)])),
        "max_depth": sorted(set([max(3, p["max_depth"] - 1), p["max_depth"], min(12, p["max_depth"] + 1)])),
        "min_child_weight": sorted(set([max(1, p["min_child_weight"] - 2), p["min_child_weight"], p["min_child_weight"] + 2])),
        "subsample": sorted(set([max(0.6, p["subsample"] - 0.1), p["subsample"], min(1.0, p["subsample"] + 0.1)])),
        "colsample_bytree": sorted(set([max(0.6, p["colsample_bytree"] - 0.1), p["colsample_bytree"], min(1.0, p["colsample_bytree"] + 0.1)])),
        "gamma": sorted(set([max(0.0, p["gamma"] - 0.2), p["gamma"], p["gamma"] + 0.2])),
        "reg_alpha": sorted(set([max(0.0, p["reg_alpha"] * 0.5), p["reg_alpha"], p["reg_alpha"] * 1.5 + 1e-12])),
        "reg_lambda": sorted(set([max(0.1, p["reg_lambda"] * 0.7), p["reg_lambda"], p["reg_lambda"] * 1.3])),
        "max_delta_step": sorted(set([max(0, p["max_delta_step"] - 1), p["max_delta_step"], p["max_delta_step"] + 1])),
    }

    # Use ParameterSampler + direct CV-like scoring on one holdout split for speed.
    x_tr, x_va, y_tr, y_va = train_test_split(
        x_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train
    )
    best_score = -1.0
    best_model = None
    sampled = list(
        ParameterSampler(
            stage2_space, n_iter=n_iter_stage2, random_state=random_state
        )
    )
    print(f"Running stage 2 tuning with {len(sampled)} candidates ...")
    for i, params in enumerate(sampled, start=1):
        candidate = XGBClassifier(**base_model.get_params())
        candidate.set_params(**params)
        candidate.set_params(early_stopping_rounds=60)
        sw = compute_sample_weight(class_weight="balanced", y=y_tr)
        candidate.fit(
            x_tr,
            y_tr,
            sample_weight=sw,
            eval_set=[(x_va, y_va)],
            verbose=False,
        )
        score = top3_scorer(candidate, x_va, y_va)
        if score > best_score:
            best_score = score
            best_model = candidate
        if i % 5 == 0 or i == len(sampled):
            print(f"  stage2 {i}/{len(sampled)} best Top-3 so far: {best_score:.4f}")

    assert best_model is not None
    print(f"Best holdout Top-3 (stage 2): {best_score:.4f}")
    return best_model


def evaluate(
    model: XGBClassifier,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    class_names: list[str],
) -> dict:
    probs = model.predict_proba(x_test)
    preds = np.argmax(probs, axis=1)

    top1 = accuracy_score(y_test, preds)
    top3 = top_k_accuracy_score(y_test, probs, k=3, labels=list(range(len(class_names))))
    report = classification_report(y_test, preds, target_names=class_names, digits=4)
    cm = confusion_matrix(y_test, preds, labels=list(range(len(class_names))))

    print("\n=== XGBoost Genre Classifier ===")
    print(f"Top-1 Accuracy: {top1:.4f}")
    print(f"Top-3 Accuracy: {top3:.4f}")
    print("\nClassification report:")
    print(report)

    return {
        "top1_accuracy": float(top1),
        "top3_accuracy": float(top3),
        "confusion_matrix": cm,
    }


def save_outputs(
    out_dir: Path,
    model: XGBClassifier,
    eval_out: dict,
    class_names: list[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    cm_df = pd.DataFrame(
        eval_out["confusion_matrix"],
        index=class_names,
        columns=class_names,
    )
    cm_df.to_csv(out_dir / "confusion_matrix.csv")

    feature_importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    feature_importance.to_csv(out_dir / "feature_importance.csv", index=False)

    metrics = {
        "top1_accuracy": eval_out["top1_accuracy"],
        "top3_accuracy": eval_out["top3_accuracy"],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved outputs to: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="High-performance genre pipeline with top-3 evaluation.",
    )
    parser.add_argument("--data", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--tune", action="store_true", help="Run randomized hyperparameter tuning")
    parser.add_argument("--n-iter-stage1", type=int, default=30, help="Stage 1 RandomizedSearch iterations")
    parser.add_argument("--n-iter-stage2", type=int, default=20, help="Stage 2 local search iterations")
    parser.add_argument("--early-stopping-rounds", type=int, default=80)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="model_outputs/high_performance_xgb",
        help="Directory for confusion matrix, metrics, and feature importance",
    )
    args = parser.parse_args()

    data = load_clean_data(args.data)
    x = data[FEATURE_COLUMNS]
    y_raw = data[CLASS_TARGET]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    class_names = list(encoder.classes_)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )
    print(f"Train: {len(x_train)} | Test: {len(x_test)} | Classes: {len(class_names)}")

    model = build_model(args.random_state)
    if args.tune:
        model = tune_model(
            base_model=model,
            x_train=x_train,
            y_train=y_train,
            random_state=args.random_state,
            n_iter_stage1=args.n_iter_stage1,
            n_iter_stage2=args.n_iter_stage2,
        )

    # Final training with early stopping on a validation split.
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train, y_train, test_size=0.15, random_state=args.random_state, stratify=y_train
    )
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_tr)
    model.set_params(early_stopping_rounds=args.early_stopping_rounds)
    model.fit(
        x_tr,
        y_tr,
        sample_weight=sample_weight,
        eval_set=[(x_val, y_val)],
        verbose=False,
    )

    eval_out = evaluate(model, x_test, y_test, class_names)
    save_outputs(Path(args.out_dir), model, eval_out, class_names)


if __name__ == "__main__":
    main()
