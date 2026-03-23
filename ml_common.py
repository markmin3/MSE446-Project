from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_DATASET = "spotify_balanced.csv"

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


def load_clean_data(path: str) -> pd.DataFrame:
    required = FEATURE_COLUMNS + [CLASS_TARGET, REGRESSION_TARGET]
    df = pd.read_csv(path)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    data = df[required].copy()
    if data["explicit"].dtype == bool:
        data["explicit"] = data["explicit"].astype(int)
    else:
        data["explicit"] = (
            data["explicit"].astype(str).str.lower().map({"true": 1, "false": 0})
        )
        data["explicit"] = data["explicit"].fillna(0).astype(int)
    return data.dropna()


def split_classification(
    data: pd.DataFrame,
    test_size: float,
    random_state: int,
):
    x = data[FEATURE_COLUMNS]
    y = data[CLASS_TARGET]
    return train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )


def split_regression(
    data: pd.DataFrame,
    test_size: float,
    random_state: int,
):
    x = data[FEATURE_COLUMNS]
    y = data[REGRESSION_TARGET]
    return train_test_split(x, y, test_size=test_size, random_state=random_state)
