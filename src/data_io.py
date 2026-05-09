"""Közös adatbetöltés. Mindenki ezt használja, hogy ugyanazt a splitet kapja."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR, DATA_VERSION, RANDOM_SEED, SAMPLE_SIZE, SPLITS_DIR, TARGET_COLUMN


def load_v1_data(
    sample_size: int | None = None,
    random_state: int = RANDOM_SEED,
    version: str = DATA_VERSION,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Betölti a final pipeline kimenetét (X_*_{version}.npy + y_*_{version}.csv).

    A fájlokat a DATA_DIR-ben keresi, illetve fallback-ként a DATA_DIR/splits-ben.

    Args:
        sample_size: ha nem None, ennyi sort vesz a train-ből, és sample_size//4-et
            a test-ből.
        random_state: a sample-eléshez.
        version: "v0" (sample) vagy "v1" (teljes adat). Default a config DATA_VERSION.

    Returns:
        (X_train, X_test, y_train, y_test, feature_names)
    """
    def _find(name: str) -> Path:
        for d in (DATA_DIR, DATA_DIR / "splits"):
            p = d / name
            if p.exists():
                return p
        raise FileNotFoundError(
            f"{name} sehol nincs (próbáltam: {DATA_DIR} és {DATA_DIR}/splits). "
            f"Futtasd a src/geostat_elemzes.py-t DATA_VERSION='{version}'-vel."
        )

    X_train = np.load(_find(f"X_train_{version}.npy"))
    X_test = np.load(_find(f"X_test_{version}.npy"))
    y_train = pd.read_csv(_find(f"y_train_{version}.csv")).iloc[:, 0].to_numpy()
    y_test = pd.read_csv(_find(f"y_test_{version}.csv")).iloc[:, 0].to_numpy()

    with open(_find(f"feature_names_{version}.json"), encoding="utf-8") as fh:
        feature_names = json.load(fh)

    if X_train.shape[1] != len(feature_names):
        raise ValueError(
            f"X_train oszlopszám ({X_train.shape[1]}) nem egyezik a "
            f"feature-lista méretével ({len(feature_names)}). "
            f"Generáld újra a verziót: python -m src.geostat_elemzes --version {version}"
        )

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
