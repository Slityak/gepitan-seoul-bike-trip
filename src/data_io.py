"""Közös adatbetöltés. Mindenki ezt használja, hogy ugyanazt a splitet kapja."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR, DATA_VERSION, RANDOM_SEED, SAMPLE_SIZE, SPLITS_DIR, TARGET_COLUMN


# A v1 final pipeline oszlopainak nevei sorrendben (a npy nem hordoz oszlopneveket).
# A `data_pipeline_final.joblib` (geostat_elemzes.feature_engineering_transform +
# VarianceThreshold) determinisztikus kimenete, 36 oszlop.
FEATURE_NAMES: list[str] = [
    "Unnamed: 0", "PLong", "PLatd", "DLong", "DLatd", "Haversine",
    "Pmonth", "Pday", "Phour",
    "Temp", "Precip", "Wind", "Humid", "Solar", "Snow", "Dust",
    "manhattan_dist", "is_same_station",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "is_rush_hour", "is_weekend",
    "day_0", "day_1", "day_2", "day_3", "day_4", "day_5", "day_6",
    "tod_night", "tod_morning", "tod_afternoon", "tod_evening", "tod_late",
]


def load_v1_data(
    sample_size: int | None = None,
    random_state: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Betölti a v1 final pipeline kimenetét (X_*_final.npy + y_*_final.csv).

    A fájlokat a DATA_DIR-ben keresi, illetve fallback-ként a DATA_DIR/splits-ben.

    Args:
        sample_size: ha nem None, ennyi sort vesz a train-ből, és sample_size//4-et
            a test-ből.
        random_state: a sample-eléshez.

    Returns:
        (X_train, X_test, y_train, y_test, feature_names)
    """
    def _find(name: str) -> Path:
        for d in (DATA_DIR, DATA_DIR / "splits"):
            p = d / name
            if p.exists():
                return p
        raise FileNotFoundError(
            f"{name} sehol nincs (próbáltam: {DATA_DIR} és {DATA_DIR}/splits)."
        )

    X_train = np.load(_find("X_train_final.npy"))
    X_test = np.load(_find("X_test_final.npy"))
    y_train = pd.read_csv(_find("y_train_final.csv")).iloc[:, 0].to_numpy()
    y_test = pd.read_csv(_find("y_test_final.csv")).iloc[:, 0].to_numpy()

    if X_train.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"X_train oszlopszám ({X_train.shape[1]}) nem egyezik a "
            f"FEATURE_NAMES méretével ({len(FEATURE_NAMES)})."
        )

    feature_names = list(FEATURE_NAMES)

    if sample_size is not None:
        rng = np.random.default_rng(random_state)
        n_train = min(sample_size, len(X_train))
        idx = rng.choice(len(X_train), size=n_train, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
        n_test = min(max(sample_size // 4, 1), len(X_test))
        idx_t = rng.choice(len(X_test), size=n_test, replace=False)
        X_test, y_test = X_test[idx_t], y_test[idx_t]

    return X_train, X_test, y_train, y_test, feature_names


def load_train_test(
    version: str = DATA_VERSION,
    sample_size: int | None = SAMPLE_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Beolvassa a train/test parquet-eket, és visszaadja X_train, X_test, y_train, y_test-et.

    Args:
        version: "v0" vagy "v1" (lásd README adat-verziók táblázat)
        sample_size: ha nem None, ekkora random sample-t vesz mindkét halmazból
                     (gyors fejlesztéshez hasznos, végső futáshoz None)

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    train_path = SPLITS_DIR / f"train_{version}.parquet"
    test_path = SPLITS_DIR / f"test_{version}.parquet"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Nem találom az adatfájlokat: {train_path}, {test_path}.\n"
            f"Az 1. hallgatótól kérd a {version} verziót Drive-ra mentve."
        )

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    if sample_size is not None:
        train_df = train_df.sample(
            n=min(sample_size, len(train_df)), random_state=RANDOM_SEED
        )
        test_df = test_df.sample(
            n=min(sample_size // 4, len(test_df)), random_state=RANDOM_SEED
        )

    if TARGET_COLUMN not in train_df.columns:
        raise KeyError(
            f"A target oszlop ({TARGET_COLUMN}) nincs az adatban. "
            f"Elérhető oszlopok: {list(train_df.columns)}"
        )

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    return X_train, X_test, y_train, y_test
