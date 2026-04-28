"""Közös adatbetöltés. Mindenki ezt használja, hogy ugyanazt a splitet kapja."""

from __future__ import annotations

import pandas as pd

from .config import DATA_VERSION, RANDOM_SEED, SAMPLE_SIZE, SPLITS_DIR, TARGET_COLUMN


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
