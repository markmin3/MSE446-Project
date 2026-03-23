# Spotify Genre Classification Project

This project builds machine learning baselines for Spotify track genre prediction after remapping many fine-grained genres into 10 macro genres:

- Pop
- Rock
- Hip-Hop / Rap
- Electronic / Dance
- Classical / Ambient
- Country / Folk
- R&B / Soul
- Jazz
- Latin
- Metal

## Project Goal

The main objective is to train and evaluate models that can predict a track's genre from Spotify audio/metadata features, and compare model quality using consistent train/test evaluation metrics.

The project currently includes:

- Dataset remapping and balancing utilities
- Separate model scripts for linear, tree-based, kernel, and boosted approaches
- A high-performance XGBoost pipeline with Top-1 and Top-3 metrics, confusion matrix export, and feature importance output

## Repository Files

- `build_dataset.py`: remaps genres and creates processed CSV datasets
- `ml_common.py`: shared feature definitions and data split helpers
- `logistic_regression_model.py`: logistic regression classifier (genre)
- `linear_regression_model.py`: linear classifier baseline (RidgeClassifier, genre)
- `random_forest_model.py`: random forest classifier + regressor
- `svm_model.py`: SVM-RBF classifier (genre)
- `xgboost_model.py`: XGBoost regressor (popularity)
- `high_performance_genre_pipeline.py`: tuned XGBoost classifier with Top-3 evaluation

## Setup

From project root:

```bash
uv sync
```

If needed, install added ML dependencies:

```bash
uv add scikit-learn xgboost
```

## Datasets

Common dataset files:

- `spotify_kaggle_dataset.csv`: original dataset
- `spotify_remapped.csv`: remapped to 10 genres
- `spotify_balanced.csv`: balanced sample (1000 tracks per genre)

Generate remapped dataset:

```bash
uv run python build_dataset.py --remap-only --output spotify_remapped.csv
```

## Features Used by Models

All classification/regression scripts use the shared feature set in `ml_common.py`:

- `duration_ms`
- `explicit`
- `danceability`
- `energy`
- `loudness`
- `speechiness`
- `acousticness`
- `instrumentalness`
- `liveness`
- `valence`
- `tempo`
- `time_signature`

## Run Models and Get Metrics

### 1) Logistic Regression (Genre Classification)

```bash
uv run python logistic_regression_model.py --data spotify_remapped.csv
```

Outputs:

- Accuracy
- Precision / recall / F1 per class
- Macro and weighted averages

### 2) Linear Model (Genre Classification)

```bash
uv run python linear_regression_model.py --data spotify_remapped.csv
```

Outputs:

- Accuracy
- Precision / recall / F1 per class

### 3) Random Forest (Classification + Regression)

```bash
uv run python random_forest_model.py --data spotify_remapped.csv
```

Outputs:

- Classification: Accuracy + full classification report
- Regression: RMSE, MAE, R^2 (popularity prediction)

### 4) SVM-RBF (Genre Classification)

```bash
uv run python svm_model.py --data spotify_balanced.csv
```

Outputs:

- Accuracy
- Full classification report

Note: Full `spotify_remapped.csv` can be slow for SVM-RBF.

### 5) XGBoost Regression (Popularity)

```bash
uv run python xgboost_model.py --data spotify_remapped.csv
```

Outputs:

- RMSE
- MAE
- R^2

### 6) High-Performance XGBoost Genre Pipeline (Top-1 + Top-3)

Quick run:

```bash
uv run python high_performance_genre_pipeline.py --data spotify_remapped.csv
```

Tuned run:

```bash
uv run python high_performance_genre_pipeline.py \
  --data spotify_remapped.csv \
  --tune \
  --n-iter-stage1 30 \
  --n-iter-stage2 20 \
  --early-stopping-rounds 100
```

Printed metrics:

- Top-1 accuracy
- Top-3 accuracy
- Full classification report

Saved artifacts (default `model_outputs/high_performance_xgb`):

- `metrics.json` (Top-1/Top-3)
- `confusion_matrix.csv`
- `feature_importance.csv`

## Suggested Evaluation Workflow

1. Run baseline classifiers on `spotify_balanced.csv` for fast iteration.
2. Re-run on `spotify_remapped.csv` for stronger final evaluation.
3. Use `high_performance_genre_pipeline.py` for Top-3 optimization and model artifacts.
4. Compare metrics across scripts and track best Top-1/Top-3 performance.

